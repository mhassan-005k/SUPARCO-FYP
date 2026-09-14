# FaithSR-Lite-X4 Personalized Ablation

Status: **completed mixed/negative result**. Latency and TIFF integrity passed; the registered quality-contribution rule did not.

Detailed preregistration: [`plan.md`](plan.md). Frozen outputs: [`artifacts`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/).

## Research question and connection

Personalized native-x4 Swin2SR improved fidelity but needed 3.42 seconds for one 512 tile. Its weakest profile was heavy compression. This experiment asked whether a roughly 0.06–0.12M parameter student could:

1. keep inference below one second;
2. inherit the personalized teacher's useful behavior;
3. improve heavy-compression robustness without damaging fidelity.

## Four frozen variants

| Variant | Teacher loss | Degradation FiLM and compression refiner | Observation correction | Parameters |
|---|---|---|---|---:|
| Base | No | No | No | 62,502 |
| Distilled | Yes | No | No | 62,502 |
| Compression | No | Yes | Yes | 121,469 |
| Full | Yes | Yes | Yes | 121,469 |

All other data, seed, epochs, optimizer and evaluation settings are identical.

## Human-readable forward function

Input `x` is **B RGB images × H×W pixels**, with normalized values from 0 to 1. Output is **B RGB images × 4H×4W**.

1. `base features = GELU(3×3 convolution(x))`
2. Six blocks apply pointwise 1×1 mixing, GELU, depthwise 7×7 spatial filtering, another GELU, and a residual addition.
3. Compression variants predict a degradation vector. FiLM converts it into per-channel scale and shift.
4. A compression gate controls an LR-space depthwise refiner. When estimated compression is low, its contribution is small.
5. A 48-channel convolution is rearranged by PixelShuffle(4) into RGB x4 output.
6. Compression variants add a bounded correction whose maximum strength is 0.20.
7. Final values are clipped to 0–1.

Compact formula:

`SR = clip(PixelShuffle4(reconstruction(conditioned features)) + bounded observation correction, 0, 1)`

Why these layers:

- **1×1 pointwise convolution** mixes channel information cheaply.
- **7×7 depthwise convolution** increases local spatial context without a full dense 7×7 cost.
- **GELU** provides smooth non-linear gating and stable residual learning.
- **FiLM** makes feature channels respond to estimated degradation.
- **Sigmoid gates** constrain compression strength and correction magnitude between interpretable bounds.
- **PixelShuffle(4)** gives exact native x4 output while most compute remains on the LR grid.
- **No uncertainty head:** the previous native-L2A calibration was negative.

## Training loss

All variants use robust Charbonnier reconstruction plus SSIM, gradient and frequency terms. Compression variants add re-observation and degradation-prediction terms. Distilled and Full add teacher pixel and teacher edge terms.

In words:

`total loss = weighted data fidelity + optional observation/degradation losses + optional teacher pixel/edge losses`

Compression samples receive larger reconstruction weight. The teacher is frozen and only the student is updated.

## Training and evaluation protocol

- Same native Sentinel-2 L2A corpus and complete-scene seed-0 split as the accepted v6 experiment: 39 train and 47 held-out tiles.
- Scale x4; 60 epochs; batch 8; 192×192 HR crops; AdamW; cosine learning rate; AMP.
- Personalized teacher checkpoint SHA-256: `b6ee155e167253fd9db8335338fffba342e2af1a201a1662942aeaa1c5949c7a`.
- Seven profiles × 47 images = 329 frozen paired cases per variant.
- Kaggle Tesla P100, PyTorch 2.5.1 CUDA 12.1.
- Best checkpoint: maximum validation PSNR.

## Main results

| Variant | PSNR | SSIM | False-edge | SAM | ERGAS | LPIPS | Observation L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | **23.68809** | **0.61846** | 0.12347 | 0.37636 | **19.27263** | 0.55761 | **0.01748** |
| Distilled | **23.68923** | 0.61839 | 0.12326 | 0.37625 | **19.27183** | 0.55893 | **0.01747** |
| Compression | 23.64276 | 0.61754 | 0.12242 | 0.37612 | 19.35876 | **0.54305** | 0.01804 |
| Full | 23.64512 | 0.61761 | **0.12220** | **0.37599** | 19.35022 | 0.54510 | 0.01800 |

Lower is better for false-edge, SAM, ERGAS, LPIPS and observation L1.

Versus bicubic:

- Base PSNR: −0.00365 dB; interval −0.03594 to +0.02865 (inconclusive).
- Distilled PSNR: −0.00261 dB; interval −0.03446 to +0.02925 (inconclusive).
- Compression PSNR: −0.04650 dB; interval −0.07927 to −0.01374 (worse).
- Full PSNR: −0.04427 dB; interval −0.07669 to −0.01185 (worse).
- All variants improved SSIM, LPIPS and LR re-observation versus bicubic, but none dominated the fidelity metrics.

## What personalization and compression actually contributed

Matched **Distilled minus Base**:

- PSNR +0.00104 dB; interval +0.00059 to +0.00148.
- False-edge −0.00018, but SSIM −0.000068 and LPIPS **worsened** by +0.00128.
- Effect size is too small and mixed for a meaningful distillation claim.

Matched **Compression minus Base**:

- Overall PSNR −0.04286 dB; interval −0.05702 to −0.02869.
- LPIPS improved by −0.01565.
- Heavy-compression PSNR improved by +0.01471 dB, but its interval −0.00039 to +0.02980 crosses zero.
- Overall observation L1 worsened by +0.000595. The preregistered compression rule therefore failed.

Matched **Full minus Compression**:

- PSNR +0.00223 dB and false-edge −0.00019, both tiny.
- LPIPS worsened by +0.00208.

Full minus Base improved heavy-compression PSNR by +0.01547 dB with interval +0.00088 to +0.03007, but overall PSNR fell 0.04063 dB and observation L1 worsened. This is a trade-off, not a win.

## Latency and application pipeline

| Variant | FP32 model P95 | FP16 model P95 | Result |
|---|---:|---:|---|
| Base | **23.35 ms** | **22.39 ms** | Pass |
| Distilled | 23.44 ms | 22.49 ms | Pass |
| Compression | 73.51 ms | 72.85 ms | Pass |
| Full | 73.35 ms | 72.78 ms | Pass |

Full end-to-end TIFF P95 was 482.85 ms FP32 and 399.19 ms FP16 for 512→2048 uncompressed output. Six audits preserved uint8, EPSG:32643, seven tags, RGB descriptions, mask and x4 transform; warnings were empty.

## Visual audit

Worst, median and best stress cases plus one training preview were inspected. Outputs are coherent and no obvious invented road/building geometry was observed, but they are soft and very close to bicubic. The worst Full case was 19.83 dB versus bicubic 20.41 dB. Visual plausibility does not override the negative PSNR evidence.

Saved views: [`Full TIFF pipeline preview`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/pipeline-fp32/pipeline_0-preview.png), [`Full worst stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/worst-native-x4.png), [`Full median stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/median-native-x4.png), and [`Full best stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/best-native-x4.png). Each stress PNG has a same-name JSON evidence record.

## Final decision and next direction

- **Do not promote Full as the final model.** It meets speed/format requirements but not the registered quality rule.
- **Do not claim successful distillation.** The teacher effect is statistically detectable only because rows are highly paired; it is not practically meaningful and worsens LPIPS.
- **Do not add Mamba yet.** No evidence identifies global context as the bottleneck.
- Keep Base as the lightweight latency floor and personalized Swin2SR as the quality ceiling.
- Next experiment should improve capacity/training supervision between these two points, using training-only Sentinel evidence and an untouched external test. Candidate changes must be individually ablated.

## Reproducibility and references

Kernel: `muhammadhassan2/faithsr-lite-x4-personalized-ablation`, private version 1. Execution source commit: `3556eb0`. Correct source dataset: private version 19; version 18 was an incomplete directory upload and was not used.

References: Conde et al., *Swin2SR* (2022); Shi et al., *PixelShuffle* (2016); Perez et al., *FiLM* (2017); Hendrycks and Gimpel, *GELU* (2016); Wang et al., *Real-ESRGAN* (2021); Wang et al., *SSIM* (2004).
