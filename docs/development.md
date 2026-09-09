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
  --output runs/faithsr3x
```

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
