import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationEncoder(nn.Module):
    """Predicts a compact degradation embedding from the LR image.

    The auxiliary 3-vector is intended for supervised synthetic-degradation
    experiments: [blur_severity, noise_severity, compression_severity], each
    normalized to [0, 1]. It is not required at inference time.
    """
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, embed_dim),
            nn.GELU(),
        )
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.embed(self.features(x))
        params = self.param_head(z)
        return z, params


class DegradationAdaptiveResidualBlock(nn.Module):
    """ONNX-friendly lightweight residual block; FiLM can be disabled for ablation."""
    def __init__(self, channels=48, embed_dim=64, expansion=2, use_deg=True):
        super().__init__()
        hidden = channels * expansion
        self.use_deg = use_deg
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.GELU()

        self.modulation = nn.Linear(embed_dim, channels * 2) if use_deg else None
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), channels, 1),
            nn.Sigmoid(),
        )
        # Keeps the initial network close to an identity residual stack.
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x, deg_embed):
        y = self.act(self.pw1(x))
        y = self.act(self.dw(y))
        y = self.pw2(y)

        if self.use_deg:
            gamma_beta = self.modulation(deg_embed)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            y = y * (1.0 + gamma) + beta
        y = y * self.channel_gate(y)
        return x + self.res_scale * y


class Upsampler(nn.Module):
    def __init__(self, channels=48, scale=4):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("v0.1 supports scale 2 or 4")
        layers = []
        stages = 1 if scale == 2 else 2
        for _ in range(stages):
            layers += [
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ]
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class FaithSRNetV01(nn.Module):
    """Candidate research architecture for faithful remote-sensing SISR.

    Core hypotheses to test (NOT claimed as novel yet):
      1) global degradation-conditioned residual reconstruction;
      2) lightweight CNN backbone for stable distortion-oriented SR;
      3) adaptive observation-consistency correction at the output.
    """
    def __init__(
        self,
        scale=4,
        channels=48,
        num_blocks=6,
        embed_dim=64,
        use_degradation_adaptation=True,
        use_observation_consistency=True,
    ):
        super().__init__()
        self.scale = scale
        self.use_degradation_adaptation = use_degradation_adaptation
        self.use_observation_consistency = use_observation_consistency

        need_deg = use_degradation_adaptation or use_observation_consistency
        self.deg_encoder = DegradationEncoder(3, embed_dim) if need_deg else None
        self.shallow = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            DegradationAdaptiveResidualBlock(channels, embed_dim, use_deg=use_degradation_adaptation)
            for _ in range(num_blocks)
        ])

        # Fuse intermediate block outputs rather than relying only on the last block.
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_blocks, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.upsampler = Upsampler(channels, scale)

        # Learns how strongly the LR observation should pull the output back
        # toward measurement consistency (one scalar per image, 0..1).
        self.consistency_gate = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        ) if use_observation_consistency else None

    def forward(self, lr, return_aux=False):
        if self.deg_encoder is not None:
            deg_embed, deg_params = self.deg_encoder(lr)
        else:
            deg_embed = None
            deg_params = None
        f0 = self.shallow(lr)
        f = f0
        feats = []
        for block in self.blocks:
            f = block(f, deg_embed)
            feats.append(f)
        f = self.fusion(torch.cat(feats, dim=1)) + f0

        residual = self.upsampler(f)
        base = F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        sr = base + residual

        if self.use_observation_consistency:
            alpha = self.consistency_gate(deg_embed).view(-1, 1, 1, 1)
            # Single learned-strength back-projection correction.
            sr_to_lr = F.interpolate(
                sr, size=(lr.shape[-2], lr.shape[-1]), mode="bicubic", align_corners=False
            )
            obs_error = lr - sr_to_lr
            correction = F.interpolate(
                obs_error, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
            sr = sr + alpha * correction
        else:
            alpha = None

        if return_aux:
            return sr, {
                "degradation_params": deg_params,
                "consistency_strength": None if alpha is None else alpha.flatten(1),
            }
        return sr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    variants = {
        "BASE": dict(use_degradation_adaptation=False, use_observation_consistency=False),
        "BASE+DA": dict(use_degradation_adaptation=True, use_observation_consistency=False),
        "BASE+DA+AOC": dict(use_degradation_adaptation=True, use_observation_consistency=True),
    }
    for name, kwargs in variants.items():
        model = FaithSRNetV01(scale=4, **kwargs)
        with torch.no_grad():
            y = model(x)
        print(f"{name:12s} params={count_parameters(model):,} input={tuple(x.shape)} output={tuple(y.shape)}")
