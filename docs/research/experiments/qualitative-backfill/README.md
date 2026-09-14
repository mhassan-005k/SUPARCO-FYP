# Deterministic Qualitative Evidence Backfill

## Purpose

Aggregate metrics can hide catastrophic local cases. This experiment made retained checkpoints inspectable on actual held-out images and stopped visual cherry-picking.

## Selection function

For each held-out observation, calculate:

`PSNR gain = model PSNR − bicubic PSNR`

Sort all gains, then save the worst, median and best cases. Each panel contains observed LR, bicubic, model output, held-out HR and absolute error; an uncertainty panel appears only when the model has that head. Same-name JSON records source, degradation parameters, metrics and checkpoint identity.

## Result

For Sentinel Full x3 across 1,728 held-out images:

- worst gain: −1.6344 dB;
- median gain: +1.8658 dB;
- best gain: +5.6059 dB.

The negative case is intentionally retained. Native-x4 Swin2SR and FaithSR-Lite later reuse the same evidence policy on their seven-profile stress grids.

## Decision

No experiment is accepted from an attractive sample alone. Figures must be mechanically selected and interpreted beside aggregate and paired results.

Artifacts: [`Sentinel qualitative cases`](../../../../artifacts/results/faithsr3x/sentinel_exact_v2/qualitative/) plus `best-preview.png/json` beside each training run.
