# SUPARCO FYP - Faithful 3x Satellite Image Super-Resolution

This repository contains the research, experiments, checkpoints, and supporting documents for a satellite image super-resolution system intended for SUPARCO analysts.

## Target system

The intended application accepts an arbitrary raster image, primarily GeoTIFF (`.tif`/`.tiff`), and returns a **3x spatially upscaled** image with:

- minimal hallucinated geographic detail;
- preservation of the source CRS, affine transform, band information, bit depth, and other available metadata;
- data-driven rather than purely perceptual reconstruction;
- processing latency below one second for a 512x512 input block, subject to target-hardware confirmation.

The model should therefore be **blind degradation-aware and image-adaptive**, not dependent on a known sensor. Sensor/GSD metadata may be used when available, but must remain optional.

## Current research status

Two model directions are represented in the repository:

1. A fine-tuned Swin2SR checkpoint and qualitative comparisons on remote-sensing imagery.
2. `FaithSRNetV01`, a lightweight experimental CNN with a learned degradation encoder, degradation-conditioned feature modulation, and adaptive observation-consistency correction.

`FaithSRNetV01` currently supports only 2x and 4x upscaling. Native 3x training and inference still need to be implemented and evaluated.

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
|   `-- faithsr_v01/             # Experimental architecture and trainer
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
`-- notebooks/                   # Kaggle/Jupyter notebooks should be added here
```

## Run the current experiment

From the repository root:

```powershell
python src/faithsr_v01/train_faithsr_v01.py `
  --data PATH_TO_IMAGE_FOLDER `
  --variant full `
  --scale 4 `
  --epochs 100 `
  --batch-size 8
```

Supported input extensions in the current trainer are `.tif`, `.tiff`, `.png`, `.jpg`, and `.jpeg`. The current PIL-based loader converts all inputs to 8-bit RGB and does **not** preserve GeoTIFF metadata; a Rasterio-based inference/data path is required for the final application.

## Documents and results

- Project proposal: [`docs/proposal/fyp-proposal.md`](docs/proposal/fyp-proposal.md)
- SUPARCO meeting requirements: [`docs/meetings/2026-07-30-suparco-meeting.txt`](docs/meetings/2026-07-30-suparco-meeting.txt)
- Input/output and metadata contract: [`docs/requirements/data-contract.md`](docs/requirements/data-contract.md)
- Updated implementation plan: [`docs/implementation-plan.md`](docs/implementation-plan.md)
- Development and reproduction guide: [`docs/development.md`](docs/development.md)
- Kaggle execution runbook: [`docs/kaggle-runbook.md`](docs/kaggle-runbook.md)
- Public and restricted dataset strategy: [`docs/requirements/dataset-strategy.md`](docs/requirements/dataset-strategy.md)
- Edge-AI reference paper: [`docs/references/edge-ai-sr-platform-paper.md`](docs/references/edge-ai-sr-platform-paper.md)
- Research paper outline: [`docs/research/paper-outline.md`](docs/research/paper-outline.md)
- Experiment audit log: [`docs/research/experiment-log.md`](docs/research/experiment-log.md)
- Result figures: [`artifacts/results/figures/`](artifacts/results/figures/)
- Stress-test metrics: [`artifacts/results/metrics/faithsr_da_vs_base_stress_test.csv`](artifacts/results/metrics/faithsr_da_vs_base_stress_test.csv)
- Archived training runs: [`artifacts/training_runs/archives/`](artifacts/training_runs/archives/)
- Base-vs-DA stress test: [`notebooks/faithsr_da_vs_base_stress_test.ipynb`](notebooks/faithsr_da_vs_base_stress_test.ipynb)
- Single-image checkpoint test: [`notebooks/faithsr_single_image_test.ipynb`](notebooks/faithsr_single_image_test.ipynb)

## Credentials and large files

`kaggle.json` is intentionally excluded by `.gitignore` and must never be committed. Model checkpoints (`.pth`) and run archives (`.zip`) are tracked through Git LFS.

Kaggle downloads or remote execution must only be performed after explicit approval.
