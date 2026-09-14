# Hybrid-X4 Capacity and Context Experiment

Status: **planned; architecture and acceptance rules drafted, code not yet implemented**

## Why this experiment is needed

FaithSR-Lite-X4 shows a large gap between speed and fidelity. Full needs only 73.35 ms model-only P95 and 482.85 ms end-to-end TIFF P95, but its overall PSNR is 0.04427 dB below bicubic. Personalized Swin2SR improves fidelity but needs 3,418.60 ms model-only P95. The next model must occupy the unused quality/latency space between 0.12M and 12.24M parameters.

Attaching Swin2SR after FaithSR is excluded because Swin2SR alone fails the latency requirement. The proposed hybrid is one trainable forward pass, with Swin2SR used only during training.

## Mathematical model

Input and output are

$$
\mathbf{x}\in[0,1]^{B\times C\times H\times W},
\qquad
\hat{\mathbf{y}}\in[0,1]^{B\times C\times4H\times4W}.
$$

The stem produces low-resolution features:

$$
\mathbf{f}_0=C_{3\times3}(\mathbf{x}).
$$

A local convolution branch $L_{\theta}$ preserves edges and small textures, while an efficient multi-scale context branch $M_{\phi}$ increases the receptive field:

$$
\mathbf{f}_L=L_{\theta}(\mathbf{f}_0),
\qquad
\mathbf{f}_M=M_{\phi}(\mathbf{f}_0).
$$

An input-dependent sigmoid gate selects their contribution per channel and spatial position:

$$
\mathbf{g}=\sigma\!\left(C_G[\mathbf{f}_L,\mathbf{f}_M,\mathbf{z}]\right),
$$

$$
\mathbf{f}_F=\mathbf{f}_0+(1-\mathbf{g})\odot\mathbf{f}_L+\mathbf{g}\odot\mathbf{f}_M.
$$

$\mathbf{z}$ is an optional input degradation/spectral summary. Native x4 reconstruction remains residual and bounded:

$$
\hat{\mathbf{y}}=\operatorname{clip}\!\left(
B_4(\mathbf{x})+\operatorname{PixelShuffle}_4(C_R(\mathbf{f}_F)),0,1\right).
$$

The first hybrid experiment will use observation consistency as a training loss, not as a mandatory output correction, because the previous correction improved perceptual metrics but worsened overall PSNR/observation trade-offs:

$$
\mathcal{L}_{\mathrm{obs}}=
\left\|D_{\psi}(\hat{\mathbf{y}})-\mathbf{x}\right\|_1.
$$

## Proposed layers and reasons

| Component | Proposed design | Reason |
|---|---|---|
| Band stem | $C\rightarrow48$ or $C\rightarrow64$, $3\times3$ convolution | More capacity than current 32-channel model while staying on LR grid |
| Local branch | residual pointwise/depthwise blocks, $3\times3$ and $7\times7$ kernels | Preserve local edges efficiently |
| Context branch | spatially adaptive multi-scale modulation with pooled resolutions | Capture fields/roads beyond one local kernel without full Swin cost |
| Fusion gate | concatenation, $1\times1$ convolution and Sigmoid | Let input content choose local versus broader context |
| Reconstruction | $C_R: F\rightarrow16C$, PixelShuffle(4) | Exact native x4 output with HR computation deferred to the end |
| Observation operator | positive unit-sum learned kernel used in the loss | Penalize unsupported output while avoiding aggressive inference correction |

## Activation functions

$$
\operatorname{GELU}(u)=u\Phi(u)
$$

is used in feature blocks for smooth residual optimization.

$$
\sigma(u)=\frac{1}{1+e^{-u}}
$$

is used only for bounded fusion/degradation gates.

$$
\operatorname{Softmax}(u_i)=\frac{e^{u_i}}{\sum_j e^{u_j}}
$$

constrains observation-kernel coefficients to be positive and sum to one. PixelShuffle is a rearrangement, not an activation. GAN/diffusion objectives are excluded because perceptual realism can reward unsupported geographic detail.

## Proposed code structure

The code does not exist yet; names below are an implementation contract, not a claim of completed code.

| Planned class/function | Responsibility |
|---|---|
| `HybridFaithSRX4Config` | Frozen bands, channels, blocks, scale and feature switches |
| `LocalResidualBranch.forward` | Computes $\mathbf{f}_L$ |
| `MultiScaleContextBranch.forward` | Computes $\mathbf{f}_M$ using efficient pooled context |
| `HybridFusionGate.forward` | Computes $\mathbf{g}$ and $\mathbf{f}_F$ |
| `HybridFaithSRX4.forward` | Complete $F_{\Theta}(\mathbf{x})$ mapping |
| `SelectiveDistillationObjective.forward` | Ground-truth loss plus teacher guidance only where teacher evidence is better |

## Matched variants

1. H0: medium CNN control without context branch.
2. H1: H0 plus multi-scale context and gated fusion.
3. H2: H1 plus selective feature/relational distillation from personalized Swin2SR.
4. H3: H2 with the audited RGB+NIR or multispectral stem, evaluated separately from RGB.

Only one registered component changes at each step.

## Latency budget

Current Full provides the measured reference:

$$
t_{\mathrm{model},95}=73.35\ \mathrm{ms},
\qquad
t_{\mathrm{TIFF},95}=482.85\ \mathrm{ms}.
$$

Subtracting percentiles is only a rough engineering estimate, but it suggests approximately 410 ms is currently spent outside the model. The proposed safety gates are:

$$
t_{\mathrm{model},95}\leq300\ \mathrm{ms},
\qquad
t_{\mathrm{TIFF},95}\leq800\ \mathrm{ms},
$$

leaving about 200 ms margin below the one-second requirement. Every gate must be remeasured end-to-end; parameter count or FLOPs cannot substitute for latency.

## Acceptance rule

- practically meaningful paired PSNR improvement over both bicubic and H0;
- no worse false-edge rate or LR observation error;
- LPIPS and spectral SAM/ERGAS reported, not optimized alone;
- model-only P95 at most 300 ms and TIFF P95 at most 800 ms on the recorded P100 protocol;
- improvement transfers from EuroSAT pretraining to untouched native-L2A scene groups;
- teacher absent from exported inference graph;
- worst/median/best result panels and exact metadata audits retained.

## Main trade-offs

A wider/deeper student increases memory and latency. Multi-scale context may reduce softness but can spread incorrect structure. Selective distillation can transfer teacher structure without runtime cost, but may also transfer teacher errors if its mask is wrong. Multispectral input can improve physical context, but makes the model Sentinel-specific and cannot be used for an arbitrary three-band TIFF.

References: Sun et al., SAFMN (ICCV 2023); Wang, EFDN (CVPRW 2022); Lanaras et al., DSen2 (2018); ESAOpenSR SEN2SR.
