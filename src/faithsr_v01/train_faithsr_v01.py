"""Train/evaluate FaithSRNet-v0.1 on a folder-style remote-sensing dataset.

Designed for UC Merced / AID-like image folders. It recursively loads:
  .tif .tiff .png .jpg .jpeg

Example:
  python train_faithsr_v01.py --data /content/UCMerced_LandUse/Images \
      --variant full --scale 4 --epochs 100 --batch-size 8

Ablations:
  --variant base   : no degradation adaptation, no observation consistency
  --variant da     : degradation adaptation only
  --variant full   : degradation adaptation + adaptive observation consistency
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import gaussian_blur

from faithsrnet_v01 import FaithSRNetV01, count_parameters


EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_roundtrip(x: torch.Tensor, quality: int) -> torch.Tensor:
    buf = BytesIO()
    tensor_to_pil(x).save(buf, format="JPEG", quality=int(quality), subsampling=0)
    buf.seek(0)
    return pil_to_tensor(Image.open(buf))


def crop_to_multiple(x: torch.Tensor, scale: int, crop_size: int | None, train: bool) -> torch.Tensor:
    _, h, w = x.shape
    max_h = h - (h % scale)
    max_w = w - (w % scale)
    if crop_size is None:
        size = min(max_h, max_w)
    else:
        size = min(crop_size, max_h, max_w)
        size -= size % scale
    if size < scale * 8:
        raise ValueError(f"Image too small after crop: {h}x{w}")

    if train:
        top = random.randint(0, max_h - size)
        left = random.randint(0, max_w - size)
    else:
        top = (max_h - size) // 2
        left = (max_w - size) // 2
    return x[:, top:top + size, left:left + size]


def augment(x: torch.Tensor) -> torch.Tensor:
    k = random.randint(0, 3)
    x = torch.rot90(x, k, dims=(1, 2))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(2,))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(1,))
    return x


def make_lr(hr: torch.Tensor, scale: int, mode: str, rng: random.Random | None = None):
    """Return (LR, normalized degradation target [blur, noise, compression])."""
    rr = rng or random
    x = hr

    if mode == "bicubic":
        lr = F.interpolate(x.unsqueeze(0), scale_factor=1 / scale, mode="bicubic", align_corners=False, antialias=True)[0]
        return lr.clamp(0, 1), torch.zeros(3, dtype=torch.float32)

    # Mixed practical synthetic degradation. Severity targets are normalized [0,1].
    blur_s = rr.random()
    noise_s = rr.random()
    comp_s = rr.random()

    sigma = 0.2 + 1.8 * blur_s
    kernel = 3 if sigma < 0.7 else (5 if sigma < 1.3 else 7)
    x = gaussian_blur(x, [kernel, kernel], [sigma, sigma])
    x = F.interpolate(x.unsqueeze(0), scale_factor=1 / scale, mode="bicubic", align_corners=False, antialias=True)[0]

    noise_std = 0.03 * noise_s
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    x = x.clamp(0, 1)

    # JPEG is optional but useful for real-world robustness; data creation is non-differentiable by design.
    quality = int(round(100 - 40 * comp_s))
    x = jpeg_roundtrip(x, quality)
    target = torch.tensor([blur_s, noise_s, comp_s], dtype=torch.float32)
    return x.clamp(0, 1), target


class RemoteSensingSRDataset(Dataset):
    def __init__(self, files: List[Path], scale=4, crop_size=256, train=True, mixed_prob=0.7, seed=42):
        self.files = files
        self.scale = scale
        self.crop_size = crop_size
        self.train = train
        self.mixed_prob = mixed_prob
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = pil_to_tensor(Image.open(self.files[idx]))
        hr = crop_to_multiple(hr, self.scale, self.crop_size, self.train)
        if self.train:
            hr = augment(hr)
            mode = "complex" if random.random() < self.mixed_prob else "bicubic"
            lr, deg = make_lr(hr, self.scale, mode)
        else:
            # Deterministic complex validation so experiments are reproducible.
            rr = random.Random(self.seed + idx)
            lr, deg = make_lr(hr, self.scale, "complex", rr)
        return lr, hr, deg


class BicubicValidationDataset(RemoteSensingSRDataset):
    def __getitem__(self, idx):
        hr = pil_to_tensor(Image.open(self.files[idx]))
        hr = crop_to_multiple(hr, self.scale, self.crop_size, False)
        lr, deg = make_lr(hr, self.scale, "bicubic")
        return lr, hr, deg


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps ** 2).mean()


def gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2 = torch.outer(g, g)
    return w2.expand(channels, 1, window_size, window_size).contiguous()


def ssim_torch(x, y, window_size=11):
    c = x.shape[1]
    window = gaussian_window(window_size, 1.5, c, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    sigma_x = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-12)
    return score.mean()


def sobel_mag(x):
    c = x.shape[1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(c, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def psnr(x, y):
    mse = F.mse_loss(x, y)
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def observation_loss(sr, lr):
    sr_down = F.interpolate(sr, size=lr.shape[-2:], mode="bicubic", align_corners=False, antialias=True)
    # Compare low-frequency content to avoid directly encouraging sensor/JPEG noise.
    a = F.avg_pool2d(sr_down, 3, stride=1, padding=1)
    b = F.avg_pool2d(lr, 3, stride=1, padding=1)
    return F.l1_loss(a, b)


def variant_kwargs(variant: str):
    if variant == "base":
        return dict(use_degradation_adaptation=False, use_observation_consistency=False)
    if variant == "da":
        return dict(use_degradation_adaptation=True, use_observation_consistency=False)
    if variant == "full":
        return dict(use_degradation_adaptation=True, use_observation_consistency=True)
    raise ValueError(variant)


def split_files(root: Path, seed=42, train_ratio=0.8):
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in EXTS])
    if len(files) < 10:
        raise RuntimeError(f"Only {len(files)} images found under {root}")
    rr = random.Random(seed)
    rr.shuffle(files)
    n = int(len(files) * train_ratio)
    return files[:n], files[n:]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs, ssims = [], []
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr).clamp(0, 1)
        psnrs.append(psnr(sr, hr).item())
        ssims.append(ssim_torch(sr, hr).item())
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["base", "da", "full"], default="full")
    ap.add_argument("--scale", type=int, choices=[2, 4], default=4)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="runs/faithsr_v01")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files = split_files(Path(args.data), args.seed)
    print(f"device={device} train={len(train_files)} val={len(val_files)}")

    train_ds = RemoteSensingSRDataset(train_files, args.scale, args.crop_size, True, seed=args.seed)
    val_complex_ds = RemoteSensingSRDataset(val_files, args.scale, args.crop_size, False, seed=args.seed)
    val_bicubic_ds = BicubicValidationDataset(val_files, args.scale, args.crop_size, False, seed=args.seed)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_complex_loader = DataLoader(val_complex_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    val_bicubic_loader = DataLoader(val_bicubic_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    model = FaithSRNetV01(scale=args.scale, **variant_kwargs(args.variant)).to(device)
    print(f"variant={args.variant} params={count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    charb = CharbonnierLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    outdir = Path(args.out) / f"{args.variant}_x{args.scale}"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "metrics.csv"
    start_epoch, best = 1, -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best_complex_psnr", -1.0)
        print(f"resumed epoch {start_epoch}")

    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "bicubic_psnr", "bicubic_ssim", "complex_psnr", "complex_ssim", "lr"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for lr, hr, deg_gt in train_loader:
            lr, hr, deg_gt = lr.to(device), hr.to(device), deg_gt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                sr, aux = model(lr, return_aux=True)
                l_pix = charb(sr, hr)
                l_ssim = 1.0 - ssim_torch(sr.float(), hr.float())
                l_grad = F.l1_loss(sobel_mag(sr.float()), sobel_mag(hr.float()))
                l_obs = observation_loss(sr.float(), lr.float())
                loss = l_pix + 0.10 * l_ssim + 0.03 * l_grad + 0.03 * l_obs
                if aux["degradation_params"] is not None:
                    l_deg = F.smooth_l1_loss(aux["degradation_params"].float(), deg_gt.float())
                    loss = loss + 0.02 * l_deg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        sched.step()
        train_loss = running / max(1, len(train_loader))
        b_psnr, b_ssim = evaluate(model, val_bicubic_loader, device)
        c_psnr, c_ssim = evaluate(model, val_complex_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        print(f"E{epoch:03d} loss={train_loss:.5f} | bicubic {b_psnr:.3f}/{b_ssim:.4f} | complex {c_psnr:.3f}/{c_ssim:.4f}")

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, b_psnr, b_ssim, c_psnr, c_ssim, lr_now])

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_complex_psnr": max(best, c_psnr),
            "args": vars(args),
        }
        torch.save(state, outdir / "last.pth")
        if c_psnr > best:
            best = c_psnr
            torch.save(state, outdir / "best.pth")

    print(f"done. best complex PSNR={best:.3f} dB; logs: {log_path}")


if __name__ == "__main__":
    main()
