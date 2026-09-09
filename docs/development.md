# FaithSR-3X Development and Reproduction

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[test,app,export,kaggle]"
```

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

## Evaluation and latency

```powershell
.venv\Scripts\python -m faithsr.evaluate --checkpoint runs/faithsr3x/best.pth --data DATASET
.venv\Scripts\python -m faithsr.compare --checkpoint runs/faithsr3x/best.pth --data DATASET
.venv\Scripts\python -m faithsr.benchmark --checkpoint runs/faithsr3x/best.pth --runs 100
```

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
