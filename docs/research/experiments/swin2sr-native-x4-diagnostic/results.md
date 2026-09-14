# Swin2SR Native-x4 Diagnostic: Accepted v2 Results

Status: accepted public-data validation and deployment diagnostic; not restricted SUPARCO validation

Date: 2026-09-13

Kernel: `muhammadhassan2/faithsr-swin2sr-native-x4-diagnostic`, version 2

Git source: `83dfac9`
Artifacts: `artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/`

## Why this experiment was necessary

SUPARCO's clarified requirement is at least 3x, so Swin2SR's native x4 output is valid and no x4-to-x3 resize is required. The earlier matched-x3 experiment proved that full domain fine-tuning is useful, but could not answer two product questions: whether native x4 remains faithful under different degradations, and whether a 512x512 input can be processed below one second. This pre-registered run answers those questions before adding any hybrid module.

Version 1 completed the three training epochs but failed while creating the 512x512 benchmark fixture because Rasterio exposes `Resampling.cubic`, not `Resampling.bicubic`. Its training values were retained as partial diagnostic evidence only. Version 2 used the corrected enum, passed the regression test, and completed every registered measurement.

## Model function, layers, and activations

For an LR RGB tensor (x\in[0,1]^{3\times H\times W}), the accepted candidate directly predicts

\[
\hat y_4=S_{\theta}(x)\in[0,1]^{3\times4H\times4W}.
\]

There is no output resize. All 12,239,283 parameters from pinned `caidas/swin2SR-classical-sr-x4-64` revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e` are trainable. AdamW with learning rate (10^{-5}) minimizes Smooth-L1:

\[
\mathcal L_{\mathrm{train}}=\frac{1}{N}\sum_i \operatorname{SmoothL1}(S_\theta(x_i),y_i).
\]

Smooth-L1 was used because it is less dominated by individual large residuals than MSE while still directly optimizing radiometric reconstruction. The inherited architecture contains a shallow convolutional embedding, residual Swin Transformer groups, shifted-window self-attention, LayerNorm, GELU MLPs, residual convolutions, and an x4 PixelShuffle head. GELU is retained because changing the pretrained nonlinearity would invalidate the learned transformer representation. PixelShuffle performs learned native x4 reconstruction without a post-output scale conversion.

```mermaid
flowchart LR
    X["48x48 synthetic LR"] --> C["Shallow convolution"]
    C --> W["Residual SwinV2 groups<br/>shifted-window attention"]
    W --> G["LayerNorm + GELU MLPs<br/>residual convolutions"]
    G --> P["x4 PixelShuffle head"]
    P --> Y["192x192 prediction"]
    Y --> L["Smooth-L1 vs held-out HR"]
```

This is genuine full-model Sentinel-2 personalization, but it remains a personalized pretrained baseline rather than being relabelled as the project's original architecture.

## Protocol

- Corpus: six public Sentinel-2 L2A scenes, content-screened 8-bit RGB tiles.
- Split: 39 training tiles and 47 validation tiles from three disjoint held-out scene parents.
- Training comparison: bicubic x4, frozen Swin2SR x4, and fully fine-tuned Swin2SR x4 on identical generated observations.
- Stress grid: seven named profiles x 47 validation images = 329 paired cases per method.
- Metrics: PSNR, SSIM, LPIPS, false-edge rate, edge precision/recall, SAM, x4-correct ERGAS, and LR re-observation L1.
- Inference benchmark: 10 warm-ups plus 100 synchronized runs for FP32 and FP16; three disk-backed TIFF pipeline runs per precision.

## Training-distribution comparison

| Method | PSNR | SSIM | False-edge | SAM | ERGAS | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic x4 | 22.5113 | 0.52003 | 0.13219 | 0.41035 | 22.7319 | 0.70549 |
| Frozen Swin2SR x4 | 20.9711 | 0.48072 | 0.13080 | 0.45810 | 30.0838 | 0.54741 |
| Fine-tuned Swin2SR x4 | **22.7824** | **0.54833** | **0.12437** | **0.39245** | 22.8896 | **0.61495** |

Fine-tuning-minus-bicubic scene-cluster mean PSNR is +0.2583 dB with 95% CI -0.0011 to +0.5176; SSIM is +0.02981, false-edge rate -0.00846, SAM -0.01821 rad, and LPIPS -0.09187. The PSNR interval narrowly crosses zero, so the stronger conclusion comes from the pre-registered multi-profile stress set rather than this single validation mixture.

## Frozen stress-grid result

Across all 329 paired cases, fine-tuned Swin2SR improves PSNR by +0.5038 dB (95% scene-cluster CI +0.4019 to +0.6056), SSIM by +0.04228, false-edge rate by -0.01460, SAM by -0.01258 rad, ERGAS by -0.92793, LPIPS by -0.08691, and re-observation L1 by -0.00291 versus bicubic.

| Stress profile | PSNR delta vs bicubic | 95% scene-cluster CI | False-edge delta | LPIPS delta |
|---|---:|---:|---:|---:|
| Compression-heavy | +0.0193 dB | -0.0626 to +0.1012 | -0.00186 | -0.09719 |
| Noise-heavy | +0.0920 dB | +0.0387 to +0.1452 | -0.00662 | -0.07525 |
| Severe blur | +0.2320 dB | +0.1766 to +0.2874 | -0.00690 | -0.03873 |
| Aliasing-heavy | +0.4151 dB | +0.2863 to +0.5438 | -0.01150 | -0.06703 |
| Low-Nyquist MTF | +0.6161 dB | +0.4732 to +0.7589 | -0.01707 | -0.08358 |
| Mild anisotropic | +0.6429 dB | +0.4851 to +0.8007 | -0.01806 | -0.08668 |
| Clean bicubic | +1.5090 dB | +1.2328 to +1.7852 | -0.04017 | -0.15992 |

Compression-heavy input is the clear weak family: its PSNR confidence interval crosses zero, even though its LPIPS and observation consistency improve. This does not establish catastrophic hallucination, but it blocks an across-condition reconstruction-superiority claim.

## Latency and TIFF contract

| Precision | Model P95 | End-to-end TIFF P95 | Peak VRAM | Result |
|---|---:|---:|---:|---|
| FP32 | 3,418.60 ms | 3,791.37 ms | 3.238 GB | Fails <1 s gate |
| FP16 | 3,922.36 ms | 4,247.67 ms | 2.896 GB | Fails <1 s gate |

FP16 saves memory but is slower on this Tesla P100 path, so reduced precision must not be advertised as an automatic speedup. Every checked pipeline output is 2048x2048 uint8 from a 512x512 uint8 TIFF. CRS `EPSG:32643`, dataset tags, RGB band descriptions, mask, and geographic bounds are preserved; affine pixel spacing is recalculated from 3.75 to 0.9375 units for native x4 output. All six audit sidecars report no warnings.

## Saved visual evidence

- [Worst compression-heavy case](../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/diagnostic/worst-native-x4.png): model 25.29 dB versus bicubic 25.45 dB; the negative case is retained.
- [Median ranked case](../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/diagnostic/median-native-x4.png): model 27.80 dB versus bicubic 27.43 dB.
- [Best ranked case](../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/diagnostic/best-native-x4.png): model 27.06 dB versus bicubic 24.19 dB.
- [Training-distribution preview](../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/training/best-preview.png): model 24.25 dB versus bicubic 23.93 dB.

All panels were selected mechanically from saved metric deltas, not by visual preference. Labels are readable, the output and reference are non-black, and every PNG has a same-name JSON record with the source path, degradation, metrics, shapes, scale, and selection rule.

## Decision and connection to the next experiment

The personalized Swin2SR teacher is accepted as the strongest native-x4 fidelity reference, but rejected as the deployment model because its P95 is over three times the target. The measured weaknesses select two mechanisms only:

1. **Efficiency:** distil the accepted personalized teacher into a compact native-x4 student; LoRA alone would reduce trainable parameters but not the teacher's inference cost.
2. **Compression robustness:** feed the estimated compression/degradation vector into the student and apply bounded LR observation consistency, with compression-stratified training batches.

A Mamba/global-context branch is not selected now because this experiment did not identify long-range structure as the dominant failure, and a heavier branch would conflict with the measured latency gap.

The next matched ablation will compare: compact native-x4 base; base plus full-run personalized-teacher distillation; base plus compression conditioning/consistency; and the complete combination. The earlier x3 distillation result was neutral, so the new hypothesis must earn its place: it differs by using the personalized native-x4 teacher, native-x4 target alignment, full-schedule teacher supervision, and compression-stratified examples. Acceptance requires P95 below one second and a statistically supported improvement on compression-heavy fidelity without degrading overall false-edge or observation-consistency metrics.
