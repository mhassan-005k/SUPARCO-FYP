# SUPARCO SRS v1.2 Compliance Audit

Audit date: 2026-09-18

Baseline: repository state immediately before this audit

Authoritative requirement: `requirements-for-super-resolution-project-v1.2.pdf`

## Executive decision

The project is **not yet SRS v1.2 acceptance-ready**. Its strongest completed capabilities are reproducible training, experiment traceability, fast tiled GeoTIFF inference, and metadata preservation for the tested 8-bit RGB path. The largest compliance gaps are 10/12-bit radiometry, operational multispectral and panchromatic models, RMSE reporting, registration measurement, batch processing, and an independent acceptance set.

Functional-requirement status is:

| Status | Count | Requirements |
|---|---:|---|
| Pass | 3 | FR4, FR5, FR11 |
| Partial | 8 | FR1, FR2, FR3, FR6, FR7, FR8, FR10, FR12 |
| Fail | 1 | FR9 |

This is a strict audit: a feature is marked **Pass** only when the current implementation and evidence satisfy the complete v1.2 wording. A feature that works only for 8-bit RGB, only in an experiment notebook, or without an acceptance test is **Partial**.

## Status definitions

- **Pass**: implemented, reproducible, and covered by suitable evidence for the stated requirement.
- **Partial**: useful implementation exists, but scope or verification does not cover the whole requirement.
- **Fail**: required capability is absent from the operational path or has no defensible evidence.
- **Pending external validation**: code may exist, but acceptance depends on an independent dataset, target hardware, or SUPARCO review.

## Functional requirements traceability

| ID | SRS v1.2 requirement | Status | Current evidence | Gap and required closure |
|---|---|---|---|---|
| FR1 | Import and validate GeoTIFF integrity, spatial reference, bit depth, bands, and dimensions | **Partial** | `src/faithsr/geotiff.py::validate_input`; corrupt-file, band, size, and geospatial tests in `tests/test_geotiff.py` and `tests/test_end_to_end.py` | The operational path deliberately rejects every dtype except `uint8`. Add explicit 8/10/12-bit interpretation, effective-bit-depth validation, and sensor/band profiles. |
| FR2 | Preserve CRS, affine transform, pixel size, extent, and projection | **Partial** | GeoTIFF writer recalculates the affine transform while retaining bounds and CRS; x3/x4 round-trip tests cover tags, descriptions, masks, and georeferencing | Evidence is strong for tested 8-bit RGB fixtures, but not yet for 10/12-bit, panchromatic, or multispectral products. Expand the verification matrix and enforce 100% pass. |
| FR3 | Preprocessing: normalization, optional contrast, registration where needed, patching, augmentation | **Partial** | `src/faithsr/data.py` provides patching and augmentation; training configs record normalization/degradation | No operational registration stage or measured registration error. Current normalization is tied to 8-bit values. Optional contrast handling is not a documented reproducible profile. |
| FR4 | Paired LR/HR data, train/validation/test splits, configurable scale and patch size | **Pass** | Deterministic paired degradation, scene/geographic split manifests, scale and crop configuration in `src/faithsr/data.py` and training pipelines | Public pairs are synthetic; this limits scientific claims but does not negate the implemented data-management function. A real registered holdout remains necessary for final acceptance. |
| FR5 | Configurable training, checkpoints, monitoring, and metrics | **Pass** | YAML/config-driven training, seeded runs, checkpoints, histories, previews, run manifests, and evaluation artifacts | Retain the current controls and add the v1.2 acceptance metrics to the standard report. |
| FR6 | LR-to-SR inference with geometric alignment and metadata | **Partial** | `src/faithsr/inference.py` performs tiled inference and writes a transformed GeoTIFF plus audit sidecar | Single-image path is uint8 and requires the checkpoint band count to match. The selected production candidate is RGB-only. Alignment is assumed from the input grid rather than measured against a reference. |
| FR7 | Objective, spectral, geometric, and perceptual evaluation with reports | **Partial** | PSNR, SSIM, false-edge, edge precision/recall, SAM, ERGAS, LPIPS, and uncertainty diagnostics exist in `src/faithsr/metrics.py` | SRS-defined RMSE reduction and mean registration error are absent. There is no single generated SRS acceptance scorecard. |
| FR8 | Synchronized input/reference/SR display, zoom, difference maps, metric summaries | **Partial** | Saved qualitative panels and error maps exist; `app.py` displays input, bicubic, and SR output | App lacks reference upload, synchronized pan/zoom, reference-aligned difference maps, and a complete metric summary. Its labels also assume 3x even when a selected model may be 4x. |
| FR9 | Batch-process multiple images or datasets in one execution | **Fail** | The core tiled routine handles large rasters | CLI and app accept one input/output at a time. Add directory/manifest execution, per-file isolation, resume behavior, and a batch summary. |
| FR10 | Export geospatial output and reports covering parameters, metrics, and results | **Partial** | GeoTIFF output and JSON audit sidecars are written; experiment folders retain metrics and figures | There is no unified per-run package containing resolved config, source/output hashes, all SRS metrics, registration results, and batch outcomes. Broader raster types remain unsupported. |
| FR11 | Log experiments, configurations, models, datasets, parameters, and results | **Pass** | Frozen configs, run manifests, model cards, experiment dossiers, hashes, figures, and result JSON/CSV artifacts are retained | Continue using one self-contained dossier per experiment and freeze final acceptance artifacts. |
| FR12 | Configure dataset, model, training, inference, and metrics without code edits | **Partial** | Training and much of inference are YAML/CLI configurable | App labels, accepted dtype, band policies, registration, metric selection, and batch behavior are partly hard-coded. Add validated product profiles and one resolved configuration schema. |

## Supported-data audit

| SRS data area | Status | Evidence and limitation |
|---|---|---|
| Platforms: WorldView, GeoEye, Pleiades, Sentinel-2, SPOT, PlanetScope | **Partial** | Sentinel-2 is used for research and stress testing. The other named platforms have not been validated with representative GeoTIFFs or sensor profiles. |
| 8-bit imagery | **Pass** | Input validation, normalization, inference, reconstruction, GeoTIFF export, and metadata tests exist. |
| 10-bit imagery | **Fail** | Commonly stored in `uint16`, but there is no effective 10-bit declaration, 0–1023 normalization, round-trip quantization test, or output policy. |
| 12-bit imagery | **Fail** | No effective 12-bit declaration, 0–4095 normalization, round-trip quantization test, or output policy. |
| RGB | **Pass** | Current checkpoints and application contract support three-band RGB. |
| Panchromatic | **Fail** | No validated one-band checkpoint, product profile, or representative sensor data. Sentinel-2 does not provide a panchromatic band. |
| Multispectral | **Fail** | The linked EuroSAT/Sentinel-2 source is suitable for study, but the current selected checkpoint is RGB-only and no 4-band or 13-band contract has passed inference/export tests. |

### Critical radiometric rule

The storage dtype and effective bit depth are different concepts. A 10-bit or 12-bit product will normally be stored in a 16-bit integer container. The system must therefore carry an explicit effective bit depth or sensor product profile; it must never normalize every `uint16` image by 65535 by assumption.

For an unsigned effective bit depth `b`, the digital-number maximum is:

```text
DN_max = 2^b - 1

x_model = clip(DN_input, 0, DN_max) / DN_max

DN_output = round(clip(x_model_output, 0, 1) * DN_max)
```

The output remains in a suitable integer storage container, while its effective 8/10/12-bit meaning, no-data value, scale, offset, and relevant TIFF tags are retained or explicitly regenerated.

## Responsible-AI and scientific-integrity audit

| ID | SRS v1.2 obligation | Status | Current evidence and remaining gap |
|---|---|---|---|
| RA1 | Do not claim interpolation creates true spatial resolution | **Pass** | Project documentation distinguishes nominal model-derived GSD from newly captured measurements and labels synthetic evidence. |
| RA2 | Distinguish enlargement, enhancement, restoration, SR, and pansharpening | **Partial** | Research notes distinguish interpolation and SR, but the app/reporting vocabulary does not consistently define all five operations. |
| RA3 | Avoid unrealistic textures and artifacts | **Partial** | Fidelity losses, LPIPS, false-edge metrics, observation consistency, and qualitative panels support conservative selection. No formal artifact gate or independent supervisor sign-off exists. |
| RA4 | Preserve geometry of buildings, roads, water, and vegetation | **Partial** | Edge metrics are available, but sub-pixel registration, boundary displacement, road-width consistency, and class-specific geometry checks are absent. |
| RA5 | Preserve spectral, radiometric, geometric, and geospatial information | **Partial** | SAM/ERGAS and 8-bit metadata checks exist. 10/12-bit radiometry, multispectral contracts, and measured registration are not complete. |
| RA6 | State uncertainty and that inference is not guaranteed | **Partial** | Uncertainty experiments and risk-map support exist, but the selected model has no accepted calibrated uncertainty output and the app lacks a mandatory limitation notice. |

## Quantitative acceptance audit

The current comparable evidence is the accepted Context Fusion x4 controlled evaluation over 329 stress cases from 47 held-out validation files and three geographic clusters. It is strong diagnostic evidence, but it is not yet the frozen independent SRS acceptance test.

| Criterion | SRS threshold | Current evidence | Status |
|---|---:|---:|---|
| PSNR gain over bicubic | at least +1.5 dB | +0.0756 dB | **Fail** |
| SSIM | at least 0.90, or at least +0.03 over bicubic | 0.62945; gain +0.02395 | **Fail** |
| RMSE reduction | at least 10% | Not computed in standard evaluation | **Fail** |
| Mean registration error | at most 0.5 output pixels | Not computed | **Fail** |
| CRS/affine/pixel size/georeference preservation | 100% | All audited 8-bit RGB x4 outputs passed without warnings | **Partial / provisional pass in tested scope** |
| 512 by 512 latency | at most 5 seconds on target GPU | model P95 48.44 ms FP16; complete TIFF P95 423.41 ms | **Pass on measured Tesla P100** |
| Visual artifact control | no significant checkerboard, ringing, or unrealistic texture | Conservative comparison panels exist | **Partial; independent review pending** |

The SRS requires at least 80% of quantitative criteria. With seven listed criteria, the release gate is **at least 6 of 7 passes**. The current result is one unconditional pass, or two of seven only if metadata is counted provisionally inside its tested scope. Therefore the 80% acceptance gate **fails**.

The separate requirement to outperform interpolation in PSNR, SSIM, and at least one perceptual metric is met on the current Context Fusion stress evaluation: PSNR and SSIM improve, and LPIPS decreases by 0.06336. This relative win does not override the failed absolute SRS thresholds.

## Scale and GSD interpretation

SRS v1.2 changes the planning target from a fixed or minimum 3x product to approaching 0.3 m GSD from inputs at 0.5 m, 1 m, and 2.5 m GSD. The corresponding nominal scale ratios are:

```text
0.5 / 0.3 = 1.67x
1.0 / 0.3 = 3.33x
2.5 / 0.3 = 8.33x
```

A native 4x model produces nominal grids of 0.125 m, 0.25 m, and 0.625 m respectively. It therefore does not provide one universal path to 0.3 m. The implementation must use a declared input GSD and a validated scale/output-grid policy. The output pixel size describes a model-derived grid and must not be presented as proof of true sensor resolution.

## Final acceptance audit

| Acceptance item | Status | Decision |
|---|---|---|
| At least 80% quantitative criteria | **Fail** | Current score is below the required 6 of 7 passes. |
| Better than interpolation in PSNR, SSIM, and one perceptual metric | **Pass in current stress test** | Context Fusion beats bicubic on PSNR, SSIM, and LPIPS, but independent confirmation is still required. |
| Geospatial metadata fully preserved | **Partial** | Passed in the 8-bit RGB test matrix, not yet across the full SRS data matrix. |
| Independent supervisor sees no misleading artifacts | **Fail / pending** | No frozen blind-review record exists. |
| Reproducible code, data, and model | **Pass** | Configs, manifests, checkpoints, dossiers, and tests are retained; restricted-data reproduction will still follow access rules. |

## Release blockers in priority order

1. Implement and test effective 10/12-bit radiometric handling without losing digital-number meaning.
2. Add RMSE and reference-registration metrics, including the 0.5-pixel gate.
3. Establish explicit RGB, panchromatic, and multispectral product profiles and checkpoints.
4. Obtain a genuinely independent, registered higher-resolution reference set for defensible 0.3 m claims.
5. Improve reconstruction quality to the PSNR/SSIM/RMSE thresholds while retaining the existing latency margin.
6. Add batch processing, synchronized reference visualization, unified reports, and independent visual sign-off.

The implementation and experiment sequence for closing these items is defined in `srs-v1.2-implementation-roadmap.md`.
