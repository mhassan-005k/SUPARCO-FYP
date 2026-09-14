# Personalized Swin2SR Native-x4 Diagnostic

## Why this followed the x3 control

SUPARCO's clarified requirement is **at least x3**, so native x4 is valid and the comparison resize is unnecessary. This experiment measured the personalized teacher directly at x4, diagnosed its weak degradation family, and tested 512-block latency before adding any custom component.

Detailed frozen protocol: [`plan.md`](plan.md). Full result tables: [`results.md`](results.md).

## Forward function

Input means **B RGB tiles × H×W pixels**, values 0–1. Output means **B RGB tiles × 4H×4W**.

`native x4 output = clip(Swin2SR with adapted weights(input), 0, 1)`

Every pretrained parameter was fine-tuned for three epochs using Smooth-L1. There is no post-model resize. Layers remain shallow convolution → residual shifted-window SwinV2 groups → LayerNorm and GELU MLPs → convolution → PixelShuffle(4).

## Stress and quality result

The frozen test has 47 held-out images from three scene parents under seven fixed degradation profiles, giving 329 paired cases.

Fine-tuned minus bicubic scene-cluster changes:

- PSNR: **+0.50377 dB**, 95% interval +0.40189 to +0.60565.
- SSIM: +0.04228.
- False-edge rate: −0.01460.
- SAM: −0.01258 radians.
- Observation L1: −0.00291.
- LPIPS: −0.08691.

Compression-heavy PSNR was only +0.0193 dB and its interval crossed zero. This was the measured weakness. A retained worst compression case was −0.15923 dB below bicubic.

## Deployment and TIFF result

Tesla P100 512→2048 model-only P95 was 3,418.60 ms in FP32 and 3,922.36 ms in FP16. The one-second gate failed. Six TIFF audits preserved uint8 type, EPSG:32643 CRS, seven dataset tags, RGB band descriptions, mask, bounds and x4-recalculated transform with no warnings.

## Decision

This is the strongest personalized fidelity teacher/reference but not the deployed final model. It selected a compact native-x4 student, teacher distillation and a compression-focused mechanism. Mamba was not added because no dominant long-range failure was measured.

Artifacts: [`swin2sr_native_x4_diagnostic_v2`](../../../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/).

References: Conde et al., *Swin2SR* (2022); Shi et al., *PixelShuffle* (2016); Wang et al., *Real-ESRGAN* (2021).
