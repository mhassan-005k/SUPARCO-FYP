# Research Paper Working Outline

Status: Living document - claims remain provisional until supported by experiments

Last updated: 2026-09-09

## Working title

**FaithSR-3X: Blind Degradation-Aware and Fidelity-Constrained Super-Resolution for Satellite Imagery**

## Central research problem

How can a lightweight 3x satellite-image super-resolution model adapt to unknown input degradations while minimizing unsupported geographic detail and preserving geospatial/radiometric information?

## Candidate contribution

The proposed system combines:

1. an input-driven degradation representation;
2. degradation-conditioned feature modulation;
3. a lightweight 3x reconstruction network;
4. learned observation consistency rather than fixed bicubic back-projection;
5. uncertainty estimation for possible hallucination regions;
6. teacher-student distillation from a strong pretrained SR model;
7. metadata-preserving tiled GeoTIFF inference.

Novelty is not claimed merely from combining components. The literature review and ablation study must establish which mechanism is new and which part produces a measurable improvement.

## Evidence available so far

The first reproducible native-3x Kaggle run trained a 117,224-parameter model for 30 epochs on EuroSAT RGB. Its best epoch reached 30.6966 dB PSNR and 0.84078 SSIM on a deterministic 6,000-image class-held-out validation split. A separate matched evaluation of the saved checkpoint reached 30.6825 dB versus 22.7390 dB bicubic PSNR, with false-edge rate 0.28037 versus 0.69036. Model-only P95 latency for a 512x512 input was 54.04 ms on a Tesla P100. The uncertainty/error correlation was 0.28648, which is encouraging but does not by itself prove external calibration. These are preliminary synthetic-degradation results; component ablations, a pretrained baseline, geographic Sentinel-2 evaluation, confidence intervals, end-to-end timing, and the restricted SUPARCO test remain required before paper claims are frozen.

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
