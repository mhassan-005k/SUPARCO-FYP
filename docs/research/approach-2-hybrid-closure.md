# Closure of the Lightweight Hybrid FaithSR Research Branch

Status: Closed after completed controlled ablations

Decision date: 2026-09-19

## Decision

The lightweight custom and hybrid FaithSR branch is closed as a candidate for further model-development experiments. Its code, checkpoints, metrics, model cards, figures, and negative results remain part of the research record, but no additional Mamba, cascade, teacher-student, degradation-conditioning, uncertainty, or context-fusion module will be added to this branch.

The decision does not mean that every experiment failed. The branch produced a reproducible GeoTIFF pipeline, sub-second lightweight inference, deterministic degradation replay, controlled ablations, and several statistically consistent improvements. It is closed because the absolute reconstruction gains were insufficient under the updated SUPARCO SRS v1.2 criteria, while the strongest personalized pretrained model now satisfies the revised five-second latency requirement.

## Evidence used for closure

| Completed experiment | Principal result | Interpretation |
|---|---|---|
| FaithSRNet v0.1 Base | 24.6625 dB PSNR and 0.6502 SSIM | Valid lightweight control |
| v0.1 degradation adaptation | +0.0134 dB over Base | Positive but too small for a contribution claim |
| v0.1 fixed observation correction | -0.0677 dB below Base | Fixed bicubic observation operator rejected |
| EuroSAT native x3 | 30.6825 dB versus 22.7390 dB bicubic on matched synthetic degradation | Verified convergence and implementation, not real-sensor superiority |
| Delhi degradation adaptation | +0.02594 dB over Plain, with 95% CI +0.02112 to +0.03076 | Small, consistent component effect |
| Delhi observation consistency | -0.00826 dB versus degradation-adaptive model | Across-metric contribution rejected |
| Delhi distillation | Approximately +0.00004 dB versus the same non-distilled student | No meaningful distillation gain |
| Native L2A Plain FaithSR | -0.26044 dB versus bicubic | Custom compact model failed native-protocol fidelity |
| Native L2A Full FaithSR | -0.33530 dB versus bicubic; negative uncertainty correlation | Full personalization and calibrated-risk claims rejected |
| Personalized Swin2SR native x4 | +0.50377 dB PSNR and +0.04228 SSIM versus bicubic | Strongest completed fidelity reference |
| FaithSR-Lite-X4 distillation | +0.00104 dB versus Lite Base and worse LPIPS | Practically insignificant teacher transfer |
| FaithSR-Lite-X4 Full | -0.04427 dB versus bicubic | Meets speed but fails reconstruction rule |
| Parallel Context Fusion | +0.03654 dB over Medium Base and +0.07561 dB over bicubic | Consistent but modest gain; SRS PSNR and SSIM thresholds still fail |
| Residual Consistency Cascade | +0.01697 dB over Medium Base | Valid mechanism, weaker than Context Fusion |

The accepted Context Fusion model processed a 512 by 512 block in 48.44 ms FP16 model-only and 423.41 ms end to end through the TIFF path on a Tesla P100. Its speed is no longer the deciding advantage. SRS v1.2 permits up to five seconds for a 512 by 512 patch, while personalized Swin2SR completed the measured FP32 model path in 3.419 seconds and the full TIFF path in 3.791 seconds.

## Closed claims

The following claims must not appear as positive findings in a thesis or paper:

- teacher distillation materially improved the deployed FaithSR student;
- degradation conditioning produced a practically large fidelity gain;
- learned observation correction improved every metric;
- the uncertainty output was calibrated on native Sentinel-2 L2A data;
- the Full Lite-X4 model outperformed bicubic reconstruction;
- the Context Fusion gain satisfies the SRS v1.2 PSNR or SSIM threshold;
- nominal x3 or x4 output pixels constitute newly sensed physical resolution.

The research paper may report these mechanisms as controlled negative or mixed results. Their value is experimental attribution, not a successful final architecture claim.

## Retained contributions

The closed branch contributes:

1. a reproducible sequence of matched architecture ablations;
2. evidence that results from synthetic EuroSAT degradation do not transfer automatically to native Sentinel-2 L2A stress data;
3. evidence that generic frozen Swin2SR is inadequate but full domain personalization is useful;
4. evidence that simple output-level distillation does not close the teacher-student fidelity gap;
5. a metadata-preserving tiled GeoTIFF inference implementation;
6. latency, precision, uncertainty, downstream, and failure-case analyses;
7. a decision rule that prevents adding architecture components without a measured deficiency and matched ablation.

## Frozen artifacts

The following experiment folders are immutable evidence for the closure paper:

- `legacy-faithsr-v01-uc-merced`
- `eurosat-rgb-native-x3`
- `sentinel-delhi-ablation`
- `sentinel-distillation-x3`
- `sentinel-native-l2a-x3`
- `faithsr-lite-x4-personalized-ablation`
- `hybrid-x4-capacity-context`
- `deployment-tiff-latency`
- `precision-and-export`
- `downstream-landcover`
- `qualitative-backfill`

Corrections to transcription, formulas, or provenance may be made, but new results must be recorded under the Swin2SR continuation track rather than appended as another hybrid variant.

## Successor track

The successor is a targeted modification of the strongest completed model, personalized Swin2SR. The first bounded hypothesis is multispectral satellite adaptation through a lightweight spectral input-output adapter while retaining the pretrained SwinV2 trunk. Radiometric 8/10/12-bit handling, RMSE, registration measurement, and GeoTIFF metadata preservation are implementation and evaluation requirements shared by every future model.

The exact successor architecture and stop rules are recorded in `swin2sr-multispectral-architecture-plan.md`.
