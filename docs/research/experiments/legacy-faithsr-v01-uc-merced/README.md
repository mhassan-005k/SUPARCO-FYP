# Legacy FaithSR v0.1 on UC Merced

## Why this experiment existed

This was the first proof that a small custom network could perform satellite-image super-resolution. It compared a plain model, degradation adaptation (DA), and DA plus observation consistency. It is historical evidence, not the final Sentinel model.

## Data and function

Dataset: UC Merced Land Use, deterministic 80/20 file split, seed 42, 100 epochs, scale x4.

Plain language: an RGB batch with **B images × 3 channels × H × W pixels** enters the network and becomes **B × 3 × 4H × 4W**. The output is clipped to normalized range 0–1.

Forward function:

$$
\hat{\mathbf{y}}=\operatorname{clip}\!\left(
\operatorname{PixelShuffle}_4(C_R(F_{\theta}(\mathbf{x})))+B_4(\mathbf{x}),0,1\right).
$$

$\mathbf{x}$ is the normalized RGB input, $F_{\theta}$ is the residual feature extractor, $C_R$ produces sub-pixel channels, and $B_4$ is bicubic x4 interpolation. The implementation is `FaithSRNetV01.forward` in `src/faithsr_v01/faithsrnet_v01.py`.

The bicubic skip preserves low-frequency evidence; residual convolution blocks learn only the missing correction. GELU was selected for smooth gradients. PixelShuffle performs deterministic x4 rearrangement without a costly HR-space trunk. DA predicts degradation features and applies FiLM-like channel scale/shift so one model can react to different blur/noise conditions.

## Results

| Variant | Best epoch | Complex PSNR | Complex SSIM |
|---|---:|---:|---:|
| Base | 74 | 24.6625 | 0.6502 |
| DA | 92 | 24.6759 | 0.6520 |
| DA + observation | 87 | 24.6082 | 0.6471 |

DA improved PSNR by only 0.0134 dB over Base. Its advantage also shrank as synthetic degradation became severe. Observation consistency reduced the main score. Therefore the architecture established feasibility, but did not support a strong novelty claim.

## Connection and decision

The weak gains motivated richer degradation modelling, exact TIFF handling and stronger pretrained baselines. Later experiments moved from UC Merced RGB scenes to Sentinel-2 data with geographic separation.

Artifacts: [`faithsr_v01_base_x4.zip`](../../../../artifacts/training_runs/archives/faithsr_v01_base_x4.zip), [`faithsr_v01_da_x4.zip`](../../../../artifacts/training_runs/archives/faithsr_v01_da_x4.zip), and [`faithsr_v01_full_x4.zip`](../../../../artifacts/training_runs/archives/faithsr_v01_full_x4.zip).

References: Shi et al., *Efficient Sub-Pixel Convolutional Neural Network* (2016); Hendrycks and Gimpel, *GELU* (2016); Perez et al., *FiLM* (2017).
