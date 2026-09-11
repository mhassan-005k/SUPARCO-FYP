# FaithSR-3X Development and Reproduction

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[test,app,export,kaggle]"
```

For exact reproduction of the verified Windows environment, install the editable package with `--no-deps` and then use [`requirements-local.lock.txt`](../requirements-local.lock.txt). Successful P100 teacher runs use the separate [`requirements-kaggle.lock.txt`](../requirements-kaggle.lock.txt) because CUDA/Torch wheels are platform-specific. `pyproject.toml` retains compatible ranges for normal package installation; the lock files freeze the evidence-producing environments.

`kaggle.json` remains local and ignored. Do not copy it into notebooks, commits, archives, or output logs.

## Local verification

```powershell
.venv\Scripts\python -m pytest
```

## Training

```powershell
.venv\Scripts\python -m faithsr.train `
  --data PATH_TO_8BIT_TRAINING_IMAGES `
  --config configs/faithsr3x_base.yaml `
  --output runs/faithsr3x `
  --experiment-rationale "Why this hypothesis must be tested" `
  --previous-experiment "Prior run ID and the exact changed component"
```

Every training run automatically writes `model-card.md` beside its checkpoint. The card contains the mathematical forward function, conditional components, Mermaid architecture figure, top-level layer/parameter table, activation rationale, exact YAML, experiment purpose, and previous-run connection. Do not accept a paper experiment whose rationale/lineage fields are left generic.

Whenever validation PSNR produces a new best checkpoint, training also overwrites `best-preview.png` and `best-preview.json`. Before training, a model-independent rule ranks held-out HR samples by spatial standard deviation and fixes the median-texture sample; this avoids both a degenerate first tile and visual cherry-picking. The preview contains observed LR, bicubic 3x, FaithSR-3X, HR reference, absolute error, and uncertainty. Bicubic and FaithSR PSNR are printed directly above their panels. The JSON records both methods' full per-image metrics, their PSNR delta, source, degradation seed/parameters, selected epoch, sample index, HR standard deviation, and selection rule, so qualitative results cannot be detached from their evidence.

Every paper-facing experiment must retain this minimum artifact bundle: `best.pth`, `last.pth`, `metrics.csv`, `split.json`, `degradations.jsonl`, `run.json`, `model-card.md`, `best-preview.png`, and `best-preview.json`. Comparison runs additionally retain per-image CSV and summary JSON. Best/median/worst examples may only be selected mechanically from the frozen per-image table; visually hand-picked examples are not accepted as evidence.

## TIFF inference

```powershell
.venv\Scripts\faithsr infer `
  --input input.tiff `
  --output output_faithsr3x.tiff `
  --checkpoint runs/faithsr3x/best.pth
```

The output is accompanied by `output_faithsr3x.tiff.audit.json` containing hashes, model version, timings, shape, CRS status, and metadata handling.

## Prepare documented Sentinel-2 8-bit RGB

Aligned Level-2A B04/B03/B02 bands can be converted into the confirmed 8-bit input domain. The command writes a GeoTIFF plus a `.derivation.json` sidecar containing source hashes and the exact radiometric mapping:

```powershell
.venv\Scripts\python -m faithsr.sentinel --red B04.tif --green B03.tif --blue B02.tif --output scene_rgb8.tiff --quantification 10000 --clip-min 0 --clip-max 0.3
```

Use the product metadata for the scene-specific additive offset when it is non-zero; the default must not be assumed for every Sentinel-2 processing baseline.

Public cloud-filtered L2A COG windows can be prepared reproducibly from the Element84 Earth Search STAC API without downloading full 10980x10980 bands:

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python scripts\prepare_sentinel_l2a_stac.py `
  --output tmp\sentinel-l2a-native-v1 `
  --per-region 2 --scene-size 1152 --tile-size 192
```

The default protocol samples two scenes for Karachi, Lahore, and Islamabad. An overview block is eligible only when at least 95% of its pixels belong to accepted L2A Scene Classification Layer classes and at least 55% remain above 0.01 surface reflectance after the item-specific scale/offset; the eligible block with maximum red-channel P90-P10 range is selected. The 55% floor was calibrated from the best attainable cloud-free Lahore block (56.63%), not chosen after model evaluation. After B04/B03/B02 conversion to documented uint8, each 192x192 tile must independently contain at least 75% nonzero pixels and mean per-channel standard deviation at least 2.0. `manifest.json`, raw STAC item JSON, each scene's derivation sidecar, TIFF tags/masks, source URLs, scene IDs, acquisition times, radiometric scale/offset, thresholds, reflectance clipping, windows, CRS, transforms, hashes, and per-tile content statistics form the provenance chain. Train from the generated `patches/` directory with [`configs/sentinel_l2a_native_kaggle.yaml`](../configs/sentinel_l2a_native_kaggle.yaml).

## Evaluation and latency

```powershell
.venv\Scripts\python -m faithsr.evaluate --checkpoint runs/faithsr3x/best.pth --data DATASET
.venv\Scripts\python -m faithsr.compare --checkpoint runs/faithsr3x/best.pth --data DATASET --config configs/faithsr3x_base.yaml
.venv\Scripts\python -m faithsr.benchmark --checkpoint runs/faithsr3x/best.pth --runs 100
```

For three-band RGB paper evaluation, add `--lpips --lpips-network alex`. LPIPS is reported alongside fidelity metrics, never used alone: a perceptually attractive low score cannot override false-edge, observation-consistency, SAM, or ERGAS regressions.

Paper-facing comparison must receive the exact training YAML through `--config`. The summary records crop size, validation seed, MTF probability and clean/complex mixture probability. A comparison whose interpolation baseline disagrees across methods on the same split is a protocol failure, not a result.

Use `--precision fp16` only on CUDA. Both CLI inference and benchmarking reject FP16 on CPU. TIFF inference records precision and compression in the audit sidecar; `--compression none` trades disk size for lower write latency.

## Application

```powershell
.venv\Scripts\streamlit run app.py
```

## ONNX export

```powershell
.venv\Scripts\python -m faithsr.export --checkpoint runs/faithsr3x/best.pth
```

TensorRT conversion is environment/hardware specific and must use the verified ONNX artifact. FP16 should be validated before INT8. An INT8 model must not be accepted when its frozen fidelity thresholds regress beyond the paper protocol.

The ONNX contract uses fixed batch size 1 with dynamic height and width. This matches tiled analyst inference and keeps the image-adaptive observation operator exportable.

Export performs ONNX structural validation plus two ONNX Runtime numerical checks at different spatial shapes by default. Use `--no-verify` only when the export dependencies are deliberately unavailable. The verified EuroSAT v8 artifact is `artifacts/exports/faithsr3x_eurosat_v8.onnx`; its sidecar records SHA-256 `d7568c054b625a9cc451dd2cb3b4e5346123ddf86492a205310664aaa14b91ec`, dynamic 3x shapes, and maximum observed FP32 parity error `1.04e-5`.
