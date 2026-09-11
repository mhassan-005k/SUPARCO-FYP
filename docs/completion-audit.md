# FaithSR-3X Completion and Evidence Audit

Audit date: 2026-09-11

## Outcome

The approved public-data engineering and experiment plan has been executed. The repository contains a native-3x, uint8 TIFF/GeoTIFF pipeline that satisfies the corrected minimum-3x requirement, the custom FaithSR-3X model, pretrained-model personalization controls, fidelity evaluation, portable export, deployment measurements, an analyst application, tests, saved qualitative results, and paper-facing documentation.

The final restricted SUPARCO validation is not complete because the imagery, final band schema, evaluation pairing, target hardware, and export policy have not been supplied. This is an external-validation gate, not a missing public-data result. No SUPARCO-performance or calibrated-uncertainty claim is made.

## Requirement-to-evidence matrix

| Requirement | State | Reproducible evidence |
|---|---|---|
| Reproducible foundation | Complete | `pyproject.toml`, both lock files, central YAML configs, deterministic split/run manifests, Kaggle metadata, and provenance records |
| Minimum 3x uint8 TIFF/GeoTIFF | Complete for current native-3x path | `src/faithsr/geotiff.py`, `src/faithsr/inference.py`, `tests/test_geotiff.py`, and `tests/test_end_to_end.py`; native x4 is also requirement-compliant and is the next candidate path |
| Preserve valid metadata | Complete | CRS/bounds, transform, tags, band metadata, no-data, color interpretation, scales/offsets, compression, and masks are captured, copied or recalculated; the audit sidecar records handling and warnings |
| Native and interpolation baselines | Complete | Native 3x FaithSR/plain model, nearest/bilinear/bicubic comparisons, and common metric tables |
| Pretrained personalization | Complete | Pinned Swin2SR frozen teacher, student distillation, and full-model fine-tuning under an explicit x4-to-x3 comparison contract. The resize was for matched x3 evaluation, not a SUPARCO constraint |
| Realistic and replayable degradation | Complete | Isotropic/anisotropic/MTF blur, noise, exact decimation, quantization/compression effects, curriculum, and per-observation `degradations.jsonl` |
| Hallucination/fidelity evaluation | Complete | PSNR, SSIM, LPIPS, false-edge, edge precision/recall, observation error, SAM, ERGAS, uncertainty correlation/calibration, paired scene-cluster intervals, and a downstream probe |
| Optimization and export | Complete for accepted targets | FP32/FP16 parity and latency, disk-backed TIFF benchmarks, and verified dynamic-shape ONNX export. TensorRT is hardware-specific; INT8 was not accepted without a frozen fidelity calibration protocol |
| Analyst application | Complete | `app.py` validates/uploads TIFF, displays metadata, reports tile progress, compares original/bicubic/model output, displays risk, and downloads TIFF/risk/audit artifacts; `tests/test_app.py` smoke-tests startup |
| Tests and documentation | Complete | Automated model, TIFF, degradation, inference, comparison, export, app, qualitative, Sentinel, downstream, pretrained, and documentation tests; research methodology and experiment lineage are documented |
| Public Kaggle execution | Complete | Private source dataset plus versioned public-data kernels; accepted artifacts and SHA-256 provenance are stored under `artifacts/` |
| Research-paper evidence | Complete for public experiments | Model mathematics, layers, activations, rationale, architecture figures, hypotheses, tables, negative results, limitations, and mechanically selected previews are in `docs/research/` and model cards |
| Restricted SUPARCO validation | Externally gated | Requires approved on-site data and final written contract items listed below |

## Accepted scientific conclusions

- The custom native-x3 pipeline is an original, compact 117,224-parameter system rather than a renamed pretrained network.
- The native Sentinel-2 L2A v6 experiment is mixed/negative evidence: Plain and Full reduce LPIPS but trail bicubic PSNR on the scene-cluster protocol; Full also worsens false-edge rate slightly.
- The present uncertainty head is not calibrated on native L2A because uncertainty-error correlation is negative. Its map is exposed as risk evidence, not as a validated confidence probability.
- Fully fine-tuned pinned Swin2SR demonstrates pretrained personalization and exceeds bicubic on the same held-out native-L2A protocol, but it is about 104x larger and retains an explicit x4-to-x3 resize.
- The corrected SUPARCO requirement is at least 3x, so native Swin2SR x4 is valid. Its native-x4 latency and degradation-wise failure profile must be measured before choosing a final or hybrid architecture.
- The strongest public results do not establish performance on restricted SUPARCO imagery.

## Saved experiment evidence

- Native Sentinel-2 L2A: `artifacts/training_runs/faithsr3x/sentinel_l2a_native_v6/`
- Fully fine-tuned pretrained control: `artifacts/training_runs/faithsr3x/swin2sr_full_finetune_native_l2a_v3/`
- Deployment benchmarks: `artifacts/results/faithsr3x/deployment_benchmark_v6/`
- FP32/FP16 parity: `artifacts/results/faithsr3x/precision_benchmark_v1/`
- Downstream land-cover probe: `artifacts/results/faithsr3x/downstream_landcover_v2/`
- Deterministic best-checkpoint and worst/median/best visual examples are stored beside their metric JSON and model cards.

## Restricted-validation gate

Before on-site execution, SUPARCO must confirm:

1. band count/order and whether every band must be reconstructed;
2. ordinary TIFF versus georeferenced GeoTIFF and the no-data/mask convention;
3. whether values are raw digital numbers or display-scaled products;
4. availability and registration of paired 3x ground truth, or an approved no-reference/operational evaluation design;
5. frozen train/adaptation/test scene identifiers with geographic independence;
6. target hardware and whether the latency limit is model-only or end-to-end;
7. which code, aggregate metrics, model weights, previews, and audit files may leave the site.

The public checkpoint must be frozen before inspection of the restricted test set. If adaptation is permitted, it must use a separate adaptation split and the final holdout must remain untouched. The existing synthetic-degradation evaluator must not be presented as a real paired SUPARCO evaluation.

## Final acceptance rule

The project can be called engineering-complete on public data. It can only be called externally validated after the restricted gate is executed, output metadata is re-opened and checked, the pre-registered fidelity thresholds are evaluated on the untouched holdout, and permitted aggregate artifacts are recorded without exporting restricted imagery.
