# Experiment Log

Architecture equations, activation/layer rationale, experiment lineage, and Mermaid figures are maintained in [`model-and-experiment-methodology.md`](model-and-experiment-methodology.md). Every new run must update both documents.

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

### deployment-benchmark-v5

- Date: 2026-09-09
- Connection: extends the model-only timing from matched evaluation v3 to the complete disk-backed TIFF pipeline.
- Why necessary: the SUPARCO requirement may cover file read, tiling, inference, blending, uncertainty output, metadata/mask handling and write—not merely a CUDA forward pass.
- Fixture: deterministic 512x512, 3-band uint8, EPSG:32643, Deflate GeoTIFF; output 1536x1536.
- Hardware: Kaggle Tesla P100.
- Model-only (20 runs, 5 warm-ups): mean 52.66 ms, median 52.59 ms, P95 52.96 ms; peak allocated VRAM 3.20 GB.
- End-to-end disk-backed pipeline (3 runs): mean 1241.28 ms, median 1253.28 ms, P95 1266.34 ms.
- Mean components: read 2.63 ms, inference 78.88 ms, disk accumulation 82.53 ms, finalization/write 1077.23 ms.
- Result: provisional one-second end-to-end target **failed** by 266.34 ms at P95. The bottleneck is lossless Deflate finalization/write, not model inference.
- Decision: retain truthful failure evidence and run matched LZW/uncompressed variants; do not report a blanket “sub-second system” claim.
- Artifact: `artifacts/results/faithsr3x/deployment_benchmark_v5/latency_512.json`.

### deployment-compression-ablation-v6

- Date: 2026-09-09
- Kaggle kernel/version: `muhammadhassan2/faithsr-3x-matched-eurosat-evaluation`, version 6
- Objective: isolate TIFF finalization/write cost after v5 missed the provisional end-to-end target.
- Matched fixture/hardware: deterministic 512x512, 3-band uint8, EPSG:32643 GeoTIFF; Tesla P100; three end-to-end runs per compression.
- Input/Deflate: mean 590.20 ms, median 513.89 ms, P95 743.43 ms.
- LZW: mean 626.85 ms, median 598.31 ms, P95 697.34 ms.
- None: mean 381.45 ms, median 343.21 ms, P95 460.42 ms.
- Result: all v6 settings passed one second; uncompressed output was fastest. The v5/v6 Deflate spread demonstrates I/O timing variability, so no hardware-independent latency guarantee is claimed.
- Artifact and hashes: `artifacts/results/faithsr3x/deployment_benchmark_v6/`.

### onnx-export-eurosat-v8

- Date: 2026-09-09
- Connection/need: follows native PyTorch deployment benchmarks and proves the portable export is executable, not merely serializable.
- Checkpoint: frozen EuroSAT v8 best model.
- Export: ONNX opset 18, batch 1, dynamic height/width and exact dynamic 3x output axes.
- Verification: ONNX checker passed; ONNX Runtime CPU matched PyTorch on 64x64→192x192 and 67x69→201x207 inputs.
- Numerical parity: maximum absolute error `1.04e-5`, mean absolute error at most `7.41e-7`.
- ONNX SHA-256: `d7568c054b625a9cc451dd2cb3b4e5346123ddf86492a205310664aaa14b91ec`.
- Artifact/sidecar: `artifacts/exports/faithsr3x_eurosat_v8.onnx` and `.json`.

## Sentinel-2 Delhi geographically separated ablations

### sentinel-delhi-ablation-v1-partial

- Date: 2026-09-09
- Kaggle kernel/version: `muhammadhassan2/faithsr-3x-sentinel-delhi-ablation`, version 1
- Dataset: `rishabhsnip/earth-observation-delhi-airshed`, public CC0 8-bit Sentinel-2 RGB display tiles.
- Split: 7,488 training images/58 coordinate groups and 1,728 validation images/14 groups; 0.1-degree grid, zero group overlap, seed 42.
- Split SHA-256: `D5AAE05587C5A18383B87D37A2CDF51CAC075DBC6EB977756B71E2548BF5F864`.
- Matched training: 30 epochs, 120x120 HR crops, batch 32, Tesla P100; maximum validation PSNR checkpoint.

| Variant | Best epoch | PSNR | SSIM | False-edge | SAM | ERGAS | UQ corr. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 25 | 24.1724 | 0.65765 | 0.13127 | 0.07608 | 8.37479 | — |
| Degradation | 22 | **24.1976** | 0.65711 | **0.12943** | **0.07491** | **8.34914** | — |
| Observation | 28 | 24.1897 | **0.65796** | 0.13164 | 0.07513 | 8.35818 | — |
| Full | 26 | 24.1911 | 0.65753 | 0.13142 | 0.07520 | 8.35577 | 0.19524 |

- Bicubic matched reference: 22.3652 dB mean per-image PSNR, 0.56177 SSIM, 0.16018 false-edge, 0.16274 SAM, 10.49418 ERGAS.
- Exact-evaluation kernel/version: `muhammadhassan2/faithsr-3x-sentinel-exact-evaluation`, version 2. Values above are per-image means; v1 batch-pooled metrics remain in the raw partial artifacts only.
- Paired component results over 14 geographic groups: DA−Plain PSNR +0.02594 dB (95% CI +0.02112,+0.03076) and false-edge -0.00192 (-0.00214,-0.00169); Observation−DA PSNR -0.00826 dB (-0.01061,-0.00590) with SSIM +0.00078; Full−Observation PSNR +0.00159 dB (+0.00049,+0.00269) with SSIM -0.00041.
- Full versus bicubic: cluster-mean PSNR +1.84378 dB (95% CI +1.69044,+1.99712) and false-edge -0.02908 (-0.03162,-0.02653).
- Interpretation: DA gives a statistically consistent but practically small gain. Observation does not improve all metrics, while uncertainty adds risk output with only tiny mixed fidelity changes. The ablation rejects a blanket “all components improve quality” claim.
- Failure: after these four variants completed, the distilled model failed before epoch 1 because Torch 2.5.1 was paired with incompatible preinstalled TorchVision 0.25.0 (`torchvision::nms` missing).
- Decision: preserve all valid artifacts and failure log; pin TorchVision 0.20.1 and run the missing teacher/personalization experiment separately with per-image geographic paired confidence intervals.
- Artifacts/model cards: `artifacts/training_runs/faithsr3x/sentinel_delhi_ablation_v1_partial/`; exact per-image evaluation: `artifacts/results/faithsr3x/sentinel_exact_v2/`.

### sentinel-distillation-v1/v2/v3

- Objective: use pinned pretrained Swin2SR as a temporary frozen teacher, then personalize the custom 117,224-parameter student to Sentinel data.
- Teacher: `caidas/swin2SR-classical-sr-x4-64`, immutable revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`; x4 output explicitly antialiased-resized to x3.
- Schedule: teacher smooth-L1 weight 0.05 for epochs 1–3; teacher-free Sentinel optimization for epochs 4–30; teacher micro-batch 2.
- Version 1 issue: Kaggle rejected an errored kernel as a `kernel_source`, so its old Full checkpoint was unavailable. The four artifacts were uploaded as private dataset `muhammadhassan2/faithsr-3x-sentinel-ablation-v1-artifacts` for the corrected version.
- Version 1 failed because of an invalid Kaggle kernel source configuration; version 2 failed because teacher inference was not isolated from autograd.
- Version 3 status: complete on Tesla P100; outputs downloaded and hashed.
- Teacher: frozen `caidas/swin2SR-classical-sr-x4-64` revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`, explicitly resized from x4 to required x3.
- Student: 117,224 parameters; teacher L1 loss weight 0.05 for epochs 1–3, then data-only training through epoch 30.
- Split: 7,488 train images / 58 coordinate groups; 1,728 validation images / 14 disjoint groups.
- Distilled result: 24.19109 dB PSNR, 0.65754 SSIM, 0.13142 false-edge rate, 0.07522 rad SAM, 8.35574 ERGAS.
- Versus bicubic: +1.84381 dB PSNR (95% geographic-cluster CI +1.69050,+1.99713).
- Unadapted pretrained x4-to-x3 baseline: 19.67141 dB PSNR and 0.45800 SSIM, below bicubic at 22.36519 dB.
- Negative result: the distilled student was effectively tied with non-distilled Full v1 (PSNR difference about +0.00004 dB); pretrained supervision did not materially improve this test.
- Decision: retain the checkpoint as reproducible personalization evidence, but prefer the simpler Full checkpoint unless another frozen protocol shows a meaningful gain.
- Artifact: `artifacts/training_runs/faithsr3x/sentinel_distillation_v3/`.

### precision-benchmark-v1

- Date: 2026-09-09
- Kaggle kernel/version: `muhammadhassan2/faithsr-3x-precision-benchmark`, version 1
- Objective/connection: follow verified ONNX export with a matched CUDA FP32-vs-FP16 speed, memory and uint8-output fidelity test.
- Protocol: identical EuroSAT v8 checkpoint and 512x512 input on Tesla P100; 10 warmups + 100 model-only runs; 10 uncompressed disk-backed TIFF runs per precision.
- FP32: model P95 53.027 ms; pipeline P95 497.860 ms; peak VRAM 3,200,190,976 bytes.
- FP16: model P95 52.143 ms; pipeline P95 450.986 ms; peak VRAM 3,067,272,704 bytes.
- Output parity: maximum 1 DN, mean 0.06787 DN; metadata equal.
- Decision: expose FP16 as an optional CUDA mode. Do not accept INT8 without calibrated fidelity testing.
- Artifact: `artifacts/results/faithsr3x/precision_benchmark_v1/`.

### downstream-landcover-v2

- Date: 2026-09-09
- Kaggle kernel/version: `muhammadhassan2/faithsr-3x-downstream-landcover`, version 2
- Why/connection: follows Sentinel reconstruction ablations and tests semantic utility with a classifier independent of SR training.
- Labels: majority non-zero WorldCover class in each coordinate-centred aligned 126x126 map window.
- Probe: 53,335-parameter 3-layer GELU CNN; trained on HR only, selected on 9 calibration groups, frozen for final testing.
- Split: 49 fit groups, 9 calibration groups, 14 untouched final-test groups; 1,726 evaluated samples and 2 unknown rare-class samples explicitly excluded.
- Best probe: epoch 8; calibration accuracy 0.5790, macro-F1 0.1632.
- Final HR: accuracy 0.65469, macro-F1 0.23651.
- Bicubic: accuracy 0.65238, macro-F1 0.23543.
- FaithSR: accuracy 0.66512, macro-F1 0.24084.
- Paired FaithSR−bicubic accuracy: +0.01540 (95% geographic-cluster CI +0.00658,+0.02422).
- Limitation: weak/imbalanced patch-majority probe; result supports modest semantic retention, not land-cover state of the art or physical subpixel recovery.
- Artifacts, checkpoint, model card, history and hashes: `artifacts/results/faithsr3x/downstream_landcover_v2/`.

## Entry template

### Experiment ID

- Date:
- Git commit:
- Kaggle kernel/version:
- Objective/hypothesis:
- Dataset and immutable version:
- Train/validation/test manifest hashes:
- Model/config:
- Mathematical forward-function section/link:
- Layer and activation rationale section/link:
- Architecture/data-flow figure link:
- Connection to previous experiment:
- Why this experiment is necessary:
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
