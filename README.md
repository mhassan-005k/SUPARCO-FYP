# SUPARCO FYP - Faithful Satellite Image Super-Resolution (minimum 3x)

This repository contains the research, experiments, checkpoints, and supporting documents for a satellite image super-resolution system intended for SUPARCO analysts.

## Target system

The intended application accepts an arbitrary raster image, primarily GeoTIFF (`.tif`/`.tiff`), and returns an image with **at least 3x spatial upscaling**. The implemented FaithSR path is native 3x; a native 4x model is also requirement-compliant and does not need to be resized back to 3x.

The output must provide:

- minimal hallucinated geographic detail;
- preservation of the source CRS, affine transform, band information, bit depth, and other available metadata;
- data-driven rather than purely perceptual reconstruction;
- processing latency below one second for a 512x512 input block, subject to target-hardware confirmation.

The model should therefore be **blind degradation-aware and image-adaptive**, not dependent on a known sensor. Sensor/GSD metadata may be used when available, but must remain optional.

## Current research status

The active `faithsr` package implements a custom native-3x model with blind degradation conditioning, learned observation consistency, uncertainty output, deterministic degradation replay, documented Sentinel-2 L2A-to-uint8 derivation, 8-bit GeoTIFF metadata preservation, disk-backed tiled inference, comparison metrics, ONNX export, and a Streamlit interface. The accepted native-x4 diagnostic shows that fully fine-tuned Swin2SR is a strong fidelity reference, but it is not the deployable final model: its 512x512-to-2048x2048 model-only P95 latency is 3.419 seconds on the measured Tesla P100, above the one-second target. Heavy compression is its weakest stress family. The next research candidate is therefore a lightweight native-x4 student personalized from the fine-tuned teacher, with compression-aware fidelity conditioning. The older `FaithSRNetV01` remains an archived 2x/4x baseline.

The first preliminary native-3x EuroSAT RGB Kaggle run completed on 2026-09-09. Its 117,224-parameter model reached 30.6966 dB PSNR and 0.84078 SSIM at epoch 25 on a deterministic 6,000-image held-out split. Matched evaluation reached 30.6825 dB versus 22.7390 dB bicubic PSNR, while model-only P95 latency was 54.04 ms for 512x512 input on a Tesla P100. This is synthetic-degradation evidence only—not yet a Sentinel-2, pretrained-baseline, end-to-end deployment, or SUPARCO claim. Raw artifacts and provenance are under `artifacts/training_runs/faithsr3x/eurosat_preliminary_v8/` and `artifacts/results/faithsr3x/eurosat_matched_v3/`.

On 1,728 geographically held-out Delhi Sentinel RGB tiles, Full FaithSR reached 24.1911 dB versus bicubic 22.3652 dB and reduced false-edge rate from 0.16018 to 0.13142. The stricter native-L2A v6 test uses six public STAC scenes from Karachi, Lahore, and Islamabad, 86 content-screened tiles, and complete scene separation. It is retained mixed/negative evidence: Plain FaithSR trails bicubic PSNR by a scene-cluster mean 0.2604 dB but improves LPIPS by 0.0919; Full trails by 0.3353 dB and its uncertainty correlation is negative. In contrast, fully fine-tuning all 12.24M parameters of pinned Swin2SR improves PSNR by 0.3365 dB, SSIM by 0.02883, false-edge rate by 0.00821, and LPIPS by 0.07080 over bicubic on the identical 47-image held-out x3-comparison protocol. The accepted native-x4 v2 stress diagnostic evaluates 329 paired cases: fine-tuning improves scene-cluster mean PSNR by 0.5038 dB (95% CI 0.4019 to 0.6056), SSIM by 0.04228, false-edge rate by 0.01460, and LPIPS by 0.08691 versus bicubic. However, compression-heavy PSNR is inconclusive (+0.0193 dB; CI -0.0626 to 0.1012), and latency fails the deployment target. Restricted SUPARCO validation remains an external gate.

### FaithSRNet v0.1 ablation results

The archived runs contain 100 training epochs each. The table reports the epoch with the best complex-degradation validation PSNR.

| Variant | Best epoch | Complex PSNR | Complex SSIM | Bicubic-degradation PSNR | Bicubic-degradation SSIM |
|---|---:|---:|---:|---:|---:|
| Base | 74 | 24.6625 | 0.6502 | 26.2198 | 0.7245 |
| Degradation adaptive (DA) | 92 | 24.6759 | 0.6520 | 26.2886 | 0.7298 |
| DA + observation consistency (Full) | 87 | 24.6082 | 0.6471 | 26.2378 | 0.7269 |

These early results do not yet establish a strong improvement: DA is only marginally above the base model, while the full variant is lower on the current complex-validation split. The consistency design and evaluation protocol require further work before novelty or performance claims are made.

The degradation stress-test notebook evaluated Base and DA checkpoints on 420 validation images:

| Condition | Bicubic PSNR | Base PSNR | DA PSNR | DA gain | Base SSIM | DA SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic | 25.1951 | 26.2208 | 26.2894 | +0.0686 dB | 0.7245 | 0.7298 |
| Mild | 24.7020 | 25.6596 | 25.6974 | +0.0378 dB | 0.6970 | 0.7008 |
| Medium | 23.8123 | 24.5080 | 24.5173 | +0.0093 dB | 0.6399 | 0.6410 |
| Severe | 22.9019 | 23.3962 | 23.4029 | +0.0067 dB | 0.5871 | 0.5877 |

The DA gain is consistent but too small to support a strong contribution claim. Future experiments must improve the degradation representation and compare against stronger pretrained baselines.

## Repository layout

```text
SUPARCO-FYP/
|-- README.md
|-- .gitignore
|-- .gitattributes
|-- src/
|   |-- faithsr/                 # Active native-3x package
|   `-- faithsr_v01/             # Archived x2/x4 architecture
|-- configs/                     # Frozen YAML experiment protocols
|-- kaggle_*/                    # Reproducible private Kaggle job sources
|-- models/
|   `-- checkpoints/
|       |-- faithsr_v01/         # Base and degradation-adaptive checkpoints
|       `-- swin2sr/             # Fine-tuned Swin2SR checkpoint
|-- artifacts/
|   |-- results/
|   |   |-- figures/             # Qualitative and metric comparison figures
|   |   `-- duplicates/          # Preserved duplicate supplied artifact
|   `-- training_runs/
|       `-- archives/            # Checkpoints and metrics from complete runs
|-- data/
|   `-- samples/                 # Non-georeferenced JPEG demonstration inputs
|-- docs/
|   |-- proposal/                # FYP proposal in Markdown and PDF
|   |-- references/              # Relevant research literature
|   `-- meetings/                # SUPARCO requirement notes
|-- scripts/                     # Data preparation and evidence backfills
|-- tests/                       # Unit and end-to-end verification
`-- notebooks/                   # Historical analysis notebooks
```

## Run the active model

From the repository root:

```powershell
.venv\Scripts\python -m faithsr.train `
  --data PATH_TO_8BIT_IMAGES `
  --config configs\faithsr3x_base.yaml `
  --output runs\faithsr3x
```

Each best checkpoint saves a deterministic held-out `best-preview.png/json` with LR, bicubic 3x, FaithSR, HR, error/risk maps, exact per-image metrics and degradation provenance. TIFF inference uses Rasterio and writes an audit sidecar; see [`docs/development.md`](docs/development.md).

## Documents and results

- Project proposal: [`docs/proposal/fyp-proposal.md`](docs/proposal/fyp-proposal.md)
- SUPARCO meeting requirements: [`docs/meetings/2026-07-30-suparco-meeting.txt`](docs/meetings/2026-07-30-suparco-meeting.txt)
- Input/output and metadata contract: [`docs/requirements/data-contract.md`](docs/requirements/data-contract.md)
- Updated implementation plan: [`docs/implementation-plan.md`](docs/implementation-plan.md)
- Completion and evidence audit: [`docs/completion-audit.md`](docs/completion-audit.md)
- Development and reproduction guide: [`docs/development.md`](docs/development.md)
- Kaggle execution runbook: [`docs/kaggle-runbook.md`](docs/kaggle-runbook.md)
- Public and restricted dataset strategy: [`docs/requirements/dataset-strategy.md`](docs/requirements/dataset-strategy.md)
- Edge-AI reference paper: [`docs/references/edge-ai-sr-platform-paper.md`](docs/references/edge-ai-sr-platform-paper.md)
- Research paper outline: [`docs/research/paper-outline.md`](docs/research/paper-outline.md)
- Native-x4 Swin2SR diagnostic pre-registration: [`docs/research/swin2sr-native-x4-diagnostic-plan.md`](docs/research/swin2sr-native-x4-diagnostic-plan.md)
- Native-x4 Swin2SR accepted results and next-model decision: [`docs/research/swin2sr-native-x4-diagnostic-results.md`](docs/research/swin2sr-native-x4-diagnostic-results.md)
- FaithSR-Lite-X4 personalized compression-aware ablation pre-registration: [`docs/research/faithsr-lite-x4-ablation-plan.md`](docs/research/faithsr-lite-x4-ablation-plan.md)
- Mathematical model, layers, activations, architecture figures, and experiment lineage: [`docs/research/model-and-experiment-methodology.md`](docs/research/model-and-experiment-methodology.md)
- Individual historical and pretrained model cards: [`docs/research/model-cards/`](docs/research/model-cards/)
- Experiment audit log: [`docs/research/experiment-log.md`](docs/research/experiment-log.md)
- Result figures: [`artifacts/results/figures/`](artifacts/results/figures/)
- Stress-test metrics: [`artifacts/results/metrics/faithsr_da_vs_base_stress_test.csv`](artifacts/results/metrics/faithsr_da_vs_base_stress_test.csv)
- Archived training runs: [`artifacts/training_runs/archives/`](artifacts/training_runs/archives/)
- Base-vs-DA stress test: [`notebooks/faithsr_da_vs_base_stress_test.ipynb`](notebooks/faithsr_da_vs_base_stress_test.ipynb)
- Single-image checkpoint test: [`notebooks/faithsr_single_image_test.ipynb`](notebooks/faithsr_single_image_test.ipynb)

## Credentials and large files

`kaggle.json` is intentionally excluded by `.gitignore` and must never be committed. Model checkpoints (`.pth`) and run archives (`.zip`) are tracked through Git LFS.

Kaggle downloads or remote execution must only be performed after explicit approval.
