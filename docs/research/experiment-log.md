# Experiment Log

This is the central audit trail for model, dataset, metric, and deployment experiments. Add one entry for every meaningful run, including failed and negative experiments.

## Existing FaithSR v0.1 runs

Dataset: UC Merced Land Use

Scale: 4x

Split: deterministic file-level 80/20 split using seed 42

Training: 100 epochs

| Run | Best epoch | Complex PSNR | Complex SSIM | Interpretation |
|---|---:|---:|---:|---|
| Base | 74 | 24.6625 | 0.6502 | Reference lightweight model |
| Degradation adaptive | 92 | 24.6759 | 0.6520 | Marginal improvement |
| DA + observation consistency | 87 | 24.6082 | 0.6471 | Lower than Base; current consistency design not supported |

### Base vs DA stress test

Sample count: 420 validation images

| Condition | Base PSNR | DA PSNR | Gain | Base SSIM | DA SSIM |
|---|---:|---:|---:|---:|---:|
| Bicubic | 26.2208 | 26.2894 | +0.0686 dB | 0.7245 | 0.7298 |
| Mild | 25.6596 | 25.6974 | +0.0378 dB | 0.6970 | 0.7008 |
| Medium | 24.5080 | 24.5173 | +0.0093 dB | 0.6399 | 0.6410 |
| Severe | 23.3962 | 23.4029 | +0.0067 dB | 0.5871 | 0.5877 |

Conclusion: DA gains are consistently positive but too small for a strong contribution claim. The gain decreases as degradation becomes more severe.

## Next experiment sequence

1. Implement and unit-test native 3x upsampling.
2. Reproduce Base-v0.1 at 3x on UC Merced as a smoke test.
3. Build the Sentinel-2 Level-2A ingestion and geographically separated split manifests.
4. Establish bicubic and strong pretrained 3x baselines.
5. Replace the global three-value degradation head with a richer representation.
6. Replace fixed bicubic observation correction with a learned degradation operator.
7. Add uncertainty calibration and false-edge evaluation.
8. Train the proposed public-data model.
9. Run component ablations with matched seeds and budgets.
10. Run on-site SUPARCO evaluation after the data contract is confirmed.

## Entry template

### Experiment ID

- Date:
- Git commit:
- Kaggle kernel/version:
- Objective/hypothesis:
- Dataset and immutable version:
- Train/validation/test manifest hashes:
- Model/config:
- Initialization/checkpoint:
- Random seed:
- Hardware:
- Training duration:
- Best validation checkpoint rule:
- Test metrics and sample count:
- Latency methodology:
- Output artifact paths:
- Result:
- Problems/failures:
- Decision and next action:
