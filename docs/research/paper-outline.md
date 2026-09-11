# Research Paper Working Outline

Status: Living document - claims remain provisional until supported by experiments

Last updated: 2026-09-11

## Working title

**FaithSR-3X: Blind Degradation-Aware and Fidelity-Constrained Super-Resolution for Satellite Imagery**

## Central research problem

How can a satellite-image super-resolution model provide at least 3x enhancement under unknown input degradations while minimizing unsupported geographic detail, preserving geospatial/radiometric information, and meeting the confirmed deployment latency scope?

## Candidate contribution

Detailed mathematical definitions, layer rationale, activation choices, architecture figures, and experiment-to-experiment connections are maintained in [`model-and-experiment-methodology.md`](model-and-experiment-methodology.md) and should be promoted into the final Methodology section after the corresponding ablations complete.

The current native-3x candidate combines:

1. an input-driven degradation representation;
2. degradation-conditioned feature modulation;
3. a lightweight 3x reconstruction network;
4. learned observation consistency rather than fixed bicubic back-projection;
5. uncertainty estimation for possible hallucination regions;
6. teacher-student distillation from a strong pretrained SR model;
7. metadata-preserving tiled GeoTIFF inference.

Novelty is not claimed merely from combining components. The literature review and ablation study must establish which mechanism is new and which part produces a measurable improvement.

## Evidence available so far

The first reproducible native-3x Kaggle run trained a 117,224-parameter model for 30 epochs on EuroSAT RGB. Its best epoch reached 30.6966 dB PSNR and 0.84078 SSIM on a deterministic 6,000-image class-held-out validation split. A separate matched evaluation reached 30.6825 dB versus 22.7390 dB bicubic PSNR, with false-edge rate 0.28037 versus 0.69036.

Geographically separated Sentinel-2 Delhi evaluation now adds 1,728 per-image measurements across 14 held-out coordinate groups. Full FaithSR reached 24.1911 dB mean per-image PSNR versus 22.3652 dB bicubic; the paired group-mean difference was +1.8438 dB (95% CI +1.6904 to +1.9971), and false-edge changed by -0.02908 (-0.03162 to -0.02653). Component evidence is nuanced: degradation conditioning improved PSNR by only +0.02594 dB (+0.02112 to +0.03076), while learned observation consistency reduced PSNR by 0.00826 dB but raised SSIM by 0.00078. Full uncertainty correlation was 0.19524.

Model-only P95 latency for a 512x512 input was about 53 ms on Tesla P100. In matched end-to-end disk-backed TIFF runs, P95 was 743 ms with input/Deflate, 697 ms with LZW, and 460 ms uncompressed, though v5/v6 variance prevents a hardware-independent guarantee. Optional FP16 reduced matched uncompressed pipeline P95 from 497.86 to 450.99 ms while changing the final uint8 output by at most 1 DN and preserving metadata. A frozen HR-trained WorldCover probe gave FaithSR 66.51% accuracy / 0.24084 macro-F1 versus bicubic 65.24% / 0.23543; paired accuracy gain was +1.54 points (95% CI +0.66 to +2.42). The probe is weak and imbalanced, so this is modest semantic evidence only.

A pinned pretrained Swin2SR x4 teacher was used for three-epoch output distillation into the native-x3 FaithSR student. The personalized student reached 24.19109 dB, effectively tied with the non-distilled Full model at 24.19106 dB, while the unadapted teacher resized to x3 reached only 19.67141 dB versus bicubic at 22.36519 dB. This negative/neutral result supports the need for domain-and-scale personalization but does not establish a distillation gain. Native L2A radiometry and the restricted SUPARCO test remain required before final claims are frozen.

Qualitative evidence is generated automatically at every best checkpoint and cannot be detached from its JSON metrics/provenance. Existing Sentinel runs have deterministic previews, while Full-model worst/median/best cases are selected mechanically from all 1,728 held-out per-image PSNR deltas. The retained worst case is 1.6344 dB below bicubic, preventing cherry-picked visual claims. Exact-protocol native L2A v6 retained 86 informative tiles from six STAC scenes and used three scene-disjoint held-out clusters. Plain FaithSR traded lower PSNR (-0.26044 dB cluster mean versus bicubic) for lower LPIPS (-0.09189); Full had -0.33530 dB PSNR and -0.07370 LPIPS. Both native uncertainty variants had negative uncertainty-error correlation, so calibrated-risk and across-metric superiority claims are explicitly rejected. On the identical validation protocol, frozen Swin2SR was -1.49232 dB below bicubic, whereas full fine-tuning was +0.33646 dB PSNR, +0.02883 SSIM, -0.00821 false-edge and -0.07080 LPIPS. This supports pretrained personalization while preserving the distinction between the 12.24M x4-plus-resize baseline and the 117k custom native-x3 model.

## Research questions

- RQ1: Does degradation conditioning improve reconstruction under unseen blur, noise, aliasing, and compression?
- RQ2: Does learned observation consistency reduce unsupported edges and structures?
- RQ3: Does uncertainty predict regions with large reconstruction or structural errors?
- RQ4: Can the model process a 512x512 input block in less than one second on the agreed deployment hardware?
- RQ5: Does the method improve downstream building, road, or land-cover interpretation compared with bicubic and pretrained baselines?

## Hypotheses and acceptance rules

- H1: The proposed model exceeds the matched base model with a paired confidence interval that excludes zero on the primary fidelity metric.
- H2: It reduces re-degradation error and false-edge rate without a material PSNR/SSIM regression.
- H3: Its uncertainty score is positively associated with reconstruction error and identifies high-risk regions.
- H4: End-to-end P95 latency satisfies the confirmed SUPARCO limit.

Thresholds will be frozen before the final test-set evaluation. Selected-image improvements are not sufficient evidence.

## Planned paper structure

1. Abstract
2. Introduction and operational motivation
3. Related work
   - remote-sensing SISR
   - blind/real-world degradation
   - faithful or task-driven SR
   - parameter-efficient adaptation and distillation
   - efficient/edge SR
4. Dataset and problem formulation
5. Proposed FaithSR-3X method
6. Training losses and optimization
7. Experimental protocol
8. Quantitative results
9. Hallucination and downstream-task analysis
10. Efficiency and GeoTIFF deployment
11. Ablation study
12. Limitations and ethical/scientific-use considerations
13. Conclusion

## Required baselines

- nearest/bilinear/bicubic interpolation;
- existing pretrained SR model without adaptation;
- full fine-tuning of the selected pretrained model;
- current FaithSR Base;
- Base + degradation adaptation;
- Base + learned observation consistency;
- teacher-distilled student;
- complete proposed system;
- optimized FP16/INT8 variants where applicable.

## Required tables and figures

- dataset and split statistics;
- model parameters, FLOPs/MACs, VRAM, and latency;
- PSNR/SSIM/LPIPS and spectral metrics where applicable;
- re-degradation and false-edge results;
- downstream task results;
- component ablation;
- robustness by degradation level;
- confidence/uncertainty calibration;
- qualitative examples with error and uncertainty maps;
- GeoTIFF metadata before/after verification;
- failure cases, not only best examples.

## Paper integrity rules

- Keep training, validation, and final test data separated.
- Do not select the checkpoint or loss weights using the test set.
- Report dataset-level means, spread/confidence intervals, and sample counts.
- Keep metric implementations and border/channel conventions fixed.
- Document negative results, including the current small DA gains and weaker Full-v0.1 result.
- Do not describe nominal 3x output GSD as newly sensed physical resolution.
