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

## FaithSR-3X EuroSAT preliminary baseline

### eurosat-preliminary-v8

- Date: 2026-09-09
- Kaggle kernel/version: `muhammadhassan2/faithsr-3x-eurosat-preliminary-baseline`, version 8
- Dataset: `apollo2506/eurosat-dataset` (RGB JPEG EuroSAT mirror)
- Split: parent-class holdout, 21,000 train / 6,000 validation images (AnnualCrop and Forest), seed 42
- Split SHA-256: `89B497E26C0AD78908194AAB95EF61D26FBC739788CFD8BAB3479424C8AA8E55`
- Model: native 3x FaithSR, 32 channels, 6 adaptive blocks, 117,224 parameters
- Training: 30 epochs, batch size 64, 63x63 HR crops, Tesla P100, PyTorch 2.5.1 CUDA 12.1 wheel
- Best checkpoint rule: maximum validation PSNR
- Best epoch: 25
- Best validation: PSNR 30.6966 dB, SSIM 0.84078, false-edge rate 0.27991, observation L1 0.05491, uncertainty/error correlation 0.28643
- Final epoch: PSNR 30.6761 dB, SSIM 0.83976
- Checkpoint SHA-256: `E3C877EE006828F161ADB6926F8D71D1B5063888668FAB1EFBABF98FD17C6C8B`
- Artifacts: `artifacts/training_runs/faithsr3x/eurosat_preliminary_v8/` and `models/checkpoints/faithsr3x/eurosat_preliminary_v8_best.pth`
- Failed attempts: versions 1-7 exposed private-clone, mount-path, packaging, P100/PyTorch architecture, AMP BCE, and validation batching issues; all are retained in the audit history.
- Matched evaluation kernel/version: `muhammadhassan2/faithsr-3x-matched-eurosat-evaluation`, version 3
- Matched comparison on the exact 6,000-image split: FaithSR 30.6825 dB / 0.84060 SSIM versus bicubic 22.7390 dB / 0.55729, bilinear 23.7156 dB / 0.58633, and nearest 21.3032 dB / 0.51278.
- Fidelity comparison: FaithSR false-edge rate 0.28037 versus bicubic 0.69036; SAM 0.03489 versus 0.12514; ERGAS 3.18576 versus 7.70141. Uncertainty/error correlation was 0.28648 and calibration error 0.00520.
- Tesla P100 model-only benchmark, 512x512 to 1536x1536: mean 52.73 ms, median 52.56 ms, P95 54.04 ms over 20 measured runs after 5 warm-ups; peak allocated VRAM 3.20 GB.
- Interpretation: the model strongly exceeds interpolation under the same synthetic degradation generator and satisfies the model-only sub-second target on a P100. This does not establish end-to-end sub-second TIFF processing, superiority to a pretrained model, generalization to Sentinel-2, or performance on the restricted SUPARCO holdout.

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
