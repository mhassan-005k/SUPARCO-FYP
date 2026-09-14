# Sentinel Delhi Geographic Ablation

## Research question and lineage

EuroSAT proved the x3 code path, but class-folder splitting and JPEG data were not adequate for satellite generalization claims. This experiment asked which FaithSR components help on Sentinel-2 RGB tiles when train and validation coordinates are separated.

## Data protocol

- Source: Delhi Airshed 8-bit Sentinel-2 RGB display tiles.
- Split: 7,488 train images in 58 coordinate groups; 1,728 validation images in 14 disjoint groups.
- Geographic grid: 0.1 degree; overlap between groups: zero; seed: 42.
- Four matched variants: Plain, Degradation, Observation, and Full.

## Model function, layers and activations

Input means **B RGB tiles, each H×W pixels, normalized to 0–1**. Output is exactly **3H×3W**.

$$
\mathbf{f}_0=\operatorname{GELU}(C_{3\times3}(\mathbf{x})),
\qquad
\mathbf{f}_6=F_{6,\theta}(\mathbf{f}_0;\mathbf{z}),
$$

$$
\hat{\mathbf{y}}_0=\operatorname{PixelShuffle}_3(C_R(\mathbf{f}_6)),
\qquad
\hat{\mathbf{y}}=\operatorname{clip}(\hat{\mathbf{y}}_0+\Delta_{\mathrm{obs}},0,1).
$$

$\mathbf{x}$ is RGB input; $F_{6,\theta}$ denotes six residual blocks; $\mathbf{z}$ is optional degradation conditioning; $C_R$ produces $3\times3^2=27$ channels; and $\Delta_{\mathrm{obs}}$ is zero for variants without observation correction. Code: `FaithSR3X.forward` in `src/faithsr/model.py`.

- 3×3 convolutions capture local edges while preserving grid alignment.
- Residual connections preserve low-frequency content and stabilize training.
- GELU gives smooth non-linear gating.
- FiLM channel scale/shift lets the degradation variant react to predicted blur/noise/compression.
- PixelShuffle(3) gives exact native x3 size.
- The uncertainty head used Softplus so risk stays non-negative; later native-L2A calibration showed this head was not reliable.

Loss combined robust pixel error, structural similarity, gradient/edge error, frequency error, optional LR observation consistency, degradation prediction, and uncertainty terms. Each term was ablated rather than assumed useful.

## Accepted per-image results

| Variant | PSNR | SSIM | False-edge | SAM | ERGAS |
|---|---:|---:|---:|---:|---:|
| Plain | 24.1724 | 0.65765 | 0.13127 | 0.07608 | 8.37479 |
| Degradation | **24.1976** | 0.65711 | **0.12943** | **0.07491** | **8.34914** |
| Observation | 24.1897 | **0.65796** | 0.13164 | 0.07513 | 8.35818 |
| Full | 24.1911 | 0.65753 | 0.13142 | 0.07520 | 8.35577 |
| Bicubic | 22.3652 | 0.56177 | 0.16018 | 0.16274 | 10.49418 |

DA minus Plain PSNR was +0.02594 dB with 95% geographic-cluster interval +0.02112 to +0.03076. This is consistent but practically small. Observation traded metrics, and Full did not dominate every simpler variant.

## Failures and decision

The first combined Kaggle job failed only when it reached the distilled variant because Torch 2.5.1 and TorchVision 0.25.0 were incompatible. Valid completed variants were preserved. The result supported degradation awareness, rejected blanket claims for all components, and created the need for a separately pinned distillation run.

Artifacts: [`training variants`](../../../../artifacts/training_runs/faithsr3x/sentinel_delhi_ablation_v1_partial/) and [`exact paired evaluation`](../../../../artifacts/results/faithsr3x/sentinel_exact_v2/).

References: Perez et al., *FiLM* (2017); Wang et al., *SSIM* (2004); ESA/Copernicus Sentinel-2 documentation.
