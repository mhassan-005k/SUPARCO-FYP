# SUPARCO SRS v1.2 Implementation and Experiment Roadmap

Roadmap date: 2026-09-18

Target duration: 8–12 months

Source: SUPARCO SRS v1.2 and the baseline compliance audit

## Outcome

The target is a reproducible satellite-image SR system that accepts declared 8/10/12-bit GeoTIFF products, supports tested RGB, panchromatic, and multispectral profiles, retains geospatial and radiometric meaning, reports RMSE and registration, and passes at least six of the seven SRS quantitative criteria on an independent frozen test set.

The uploaded operational TIFF is never synthetically degraded. Degradation is used only to construct supervised training/evaluation pairs where real paired data are unavailable.

## Design principles

1. **Fidelity before visual sharpness.** GAN or diffusion losses are not introduced unless they pass the same radiometric, spectral, geometric, and artifact gates.
2. **Explicit product profiles.** Bit depth, bands, band order, scale/offset, no-data, GSD, and sensor family are declared and validated.
3. **No silent band reattachment.** Every generated band is processed by a declared model path. Untouched bands, if a product permits them, are identified as untouched.
4. **Registration before scoring.** Misregistered pairs cannot be used to claim reconstruction quality.
5. **Independent acceptance once.** Model choice stops before the frozen final set is evaluated.
6. **Nominal GSD is not captured resolution.** Reports state the input GSD, requested grid, scale, and evidence limits.

## Delivery phases

### Phase 0 — Requirements traceability and metric contract (weeks 1–2)

Deliverables:

- machine-readable SRS traceability table linked to tests and artifacts;
- RMSE and percentage-reduction implementation;
- reference-registration metric and acceptance report;
- immutable development, validation, and independent-test manifests;
- one scorecard generator with all seven quantitative criteria.

RMSE for `N` valid reference pixels is:

```text
RMSE = sqrt((1 / N) * sum((SR_i - HR_i)^2))
```

RMSE improvement over bicubic is:

```text
RMSE_reduction_percent = 100 * (RMSE_bicubic - RMSE_SR) / RMSE_bicubic
```

The pass rule is `RMSE_reduction_percent >= 10`.

For matched control points or displacement estimates `(dx_i, dy_i)`, mean registration error is:

```text
registration_error_i = sqrt(dx_i^2 + dy_i^2)

mean_registration_error = (1 / M) * sum(registration_error_i)
```

The result must be reported in output pixels and pass at `mean_registration_error <= 0.5`. Phase correlation may provide the initial global translation; feature/control-point checks and visual overlays must verify difficult scenes.

Exit gate: metric unit tests pass on known synthetic shifts and known pixel values; the generated scorecard cannot mark missing metrics as a pass.

### Phase 1 — 8/10/12-bit radiometric foundation (weeks 2–5)

Implement a `DigitalNumberCodec`-style boundary component with these fields:

```text
storage_dtype
effective_bit_depth
valid_min and valid_max
nodata
scale and offset
normalization policy
```

For unsigned imagery:

```text
DN_max = 2^effective_bit_depth - 1
x = clip(DN, 0, DN_max) / DN_max
DN_reconstructed = round(clip(x_hat, 0, 1) * DN_max)
```

Implementation requirements:

- keep 10/12-bit products in a compatible integer container such as `uint16`;
- read effective bit depth from an authoritative tag/profile or require user configuration;
- reject ambiguous `uint16` rather than guessing 10, 12, or 16 bits;
- retain no-data, masks, per-band scale/offset, units, descriptions, and relevant TIFF metadata;
- compute metrics in both normalized space and digital-number units;
- make saturation and out-of-range counts part of the audit report.

Verification matrix:

| Effective depth | Storage | Test range | Required round-trip behavior |
|---:|---|---:|---|
| 8 | uint8 | 0–255 | exact metadata; quantization error at most 0.5 DN before integer rounding |
| 10 | uint16 | 0–1023 | preserve effective-depth declaration and no-data; no 65535 normalization |
| 12 | uint16 | 0–4095 | preserve effective-depth declaration and no-data; no 65535 normalization |

Exit gate: all depths pass x2/x4/x8 GeoTIFF transform, bounds, CRS, masks, tags, saturation, and deterministic round-trip tests.

### Phase 2 — Spectral and sensor profiles (weeks 4–8)

Add validated profiles in this order:

1. **RGB, 3 bands** — maintain the existing control and make band order explicit.
2. **Panchromatic, 1 band** — train/evaluate only when representative WorldView, GeoEye, Pleiades, SPOT, or PlanetScope data are legally available. Sentinel-2 cannot validate this profile because it has no panchromatic band.
3. **Sentinel-2 RGB+NIR, 4 bands at 10 m** — first multispectral milestone using B2, B3, B4, and B8 on a common native grid.
4. **Sentinel-2 extended multispectral** — only after documenting how 20 m and 60 m bands are aligned to the chosen grid. Every resampling operation and native GSD remains in provenance.

Model input/output channels, normalization statistics, band order, wavelength identity, scale/offset, and loss weights become profile configuration rather than code edits. Spectral Angle Mapper and ERGAS are mandatory for multispectral runs; per-band RMSE/PSNR/SSIM are also reported.

Exit gate: a wrong band count/order fails before inference; each supported profile passes a full GeoTIFF read/model/write test with band metadata intact.

### Phase 3 — Dataset and quality experiments (weeks 6–16)

Data roles:

- linked EuroSAT/Sentinel-2 TIFFs: multispectral/domain pretraining and land-cover diversity;
- native Sentinel-2 L2A scenes: 512-pixel stress, out-of-distribution behavior, and metadata validation;
- UC Merced: historical RGB smoke/control only;
- registered higher-resolution or restricted SUPARCO data: final physical and operational validation.

Before use, pin dataset versions and create manifests containing file hash, source scene, geographic group, product level, shape, storage dtype, effective bit depth, bands, band GSD, CRS, transform, valid range, and quality masks. Split by source scene/geography, not by randomly sampling neighboring tiles.

Quality experiments retain Context Fusion x4 as the current fast baseline. Hybrid components are accepted only when an ablation shows which specific gap they close. The model-selection objective must include PSNR gain, SSIM, RMSE reduction, registration, SAM/ERGAS where applicable, false-edge behavior, LPIPS, latency, and memory.

Exit gate before final test: development/validation evidence reaches the SRS thresholds without regression in metadata or latency. If it does not, continue data/loss/architecture work; do not consume the independent test.

### Phase 4 — Registration and geometric validation (weeks 8–13)

Registration occurs at two levels:

- **dataset registration**: align LR/HR sources before extracting training pairs; log transforms and reject pairs that exceed the registered threshold;
- **output validation**: compare SR and reference using global phase correlation plus robust local features/control points.

Add deterministic fixtures for integer and fractional translation, rotation warning, no-overlap failure, masked regions, and texture-poor imagery. Add application-relevant geometry checks such as boundary displacement, road-width consistency, and edge precision/recall for buildings, roads, water, and vegetation.

Exit gate: mean registration error is at most 0.5 output pixels on the frozen validation protocol, and every excluded pair has a recorded reason.

### Phase 5 — Application, batch processing, and reporting (weeks 12–18)

Implement:

- multi-file upload, directory, or manifest input;
- per-file validation and failure isolation;
- resumable processing with collision-safe output names;
- reference upload and synchronized input/reference/SR pan and zoom;
- reference-aligned difference maps and per-image/aggregate metrics;
- product-profile selector for bit depth, bands, sensor, and requested output grid;
- resolved configuration, hashes, runtime, device, warnings, metadata audit, and metric scorecard in one export bundle;
- a mandatory notice that model inference is an estimate and may not represent true captured detail;
- calibrated uncertainty/risk visualization only if calibration passes a registered experiment.

Exit gate: batch tests include mixed valid/invalid files, interruption/resume, duplicate names, all supported profiles, and deterministic summary totals.

### Phase 6 — Frozen acceptance and handover (weeks 18–24)

1. Freeze code, container/environment, checkpoint, product profiles, and independent manifest.
2. Run the independent set once without tuning.
3. Generate the seven-criterion scorecard; at least six criteria must pass.
4. Conduct a blind supervisor review for checkerboard, ringing, false structures, and misleading texture.
5. Verify every output's CRS, transform, extent, pixel size, band schema, bit depth, masks, no-data, scale, and offset.
6. Archive reports, figures, logs, hashes, checkpoint, environment lock, and exact reproduction command.

If fewer than six criteria pass, the release is not accepted. Results remain valid research evidence and are reported without relabeling thresholds.

## Registered experiment sequence

Each experiment receives its own folder containing a plan, exact configuration, mathematical model, layers and activation rationale, dataset manifest, logs, checkpoints, metrics, sample outputs, failure analysis, references, and connection to the previous experiment.

| ID | Experiment | Main comparison | Pass/stop rule |
|---|---|---|---|
| E20 | Metric and registration contract | Known synthetic values/shifts vs computed RMSE and displacement | Exact/within-tolerance unit results; missing/invalid reference must fail loudly |
| E21 | Bit-depth round-trip matrix | 8 vs 10 vs 12-bit fixtures through read/infer/write | Metadata passes 100%; no effective-range corruption; quantization within declared tolerance |
| E22 | Spectral contract | RGB, 1-band PAN when data exists, 4-band Sentinel-2 RGB+NIR | Correct channel/band provenance; multispectral improves over bicubic without SAM/ERGAS regression |
| E23 | Context Fusion data-scale study | Current training vs EuroSAT TIFF pretraining plus Sentinel-2 stress | Continue only if geographically held-out fidelity improves and gains are not limited to easy bicubic degradation |
| E24 | GSD/scale policy | Native x2, x4, and x8 or equivalent validated heads | Select per-input-GSD route by fidelity, artifact, latency, and target-grid evidence; never infer physical resolution from scale alone |
| E25 | Independent SRS v1.2 acceptance | Frozen candidate vs bicubic | At least 6/7 quantitative passes, baseline superiority condition, metadata pass, and supervisor sign-off |

## E24 scale policy to test

The desired scale to a 0.3 m output grid is:

```text
requested_scale = input_GSD / 0.3
```

The three stated examples require approximately 1.67x, 3.33x, and 8.33x. Experiments should compare discrete native x2/x4/x8 routes and a declared final grid resampling step where necessary. A resampling step may provide the requested raster dimensions, but it is not counted as recovered physical detail. The x8 route receives an especially strict artifact and latency review.

## Acceptance scorecard schema

Every final report must provide:

| Field | Pass rule |
|---|---|
| PSNR gain over bicubic | at least +1.5 dB |
| SSIM | at least 0.90 or gain at least +0.03 |
| RMSE reduction | at least 10% |
| Mean registration error | at most 0.5 output pixels |
| Geospatial preservation | 100% |
| 512 by 512 latency | at most 5 seconds on declared target GPU |
| Visual artifact review | no visually significant failure |
| Overall quantitative acceptance | at least 6 of 7 criteria pass |

The report must also show PSNR, SSIM, and at least one perceptual metric relative to interpolation, plus per-band spectral metrics for multispectral profiles.

## Immediate next sprint

1. Implement E20: RMSE, registration, and scorecard tests.
2. Implement E21: replace uint8-only I/O assumptions with an explicit digital-number codec while retaining strict validation.
3. Audit representative linked EuroSAT TIFFs and freeze the 4-band Sentinel-2 profile for E22.
4. Add the independent-set manifest schema before any new tuning.
5. Keep the current Context Fusion checkpoint as the performance baseline; do not promote a new hybrid until it closes a measured SRS gap.
