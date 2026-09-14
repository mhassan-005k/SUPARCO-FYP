# Fully Fine-tuned Swin2SR, Matched x3 Comparison

## Research question

The native-L2A custom models lost PSNR. This control asked whether adapting every parameter of a strong pretrained model to the same Sentinel split fixes the domain gap. It tests real personalization rather than presenting a downloaded checkpoint as the project.

## Function in plain language

Input is **B RGB images × H×W**, normalized to 0–1. Pinned Swin2SR natively produces **4H×4W**. For comparison with the earlier x3 experiment only, a declared antialiased resize converts the output to **3H×3W**:

$$
\hat{\mathbf{y}}_3=R_{4\rightarrow3}\!\left(T_{\theta,4}(\mathbf{x})\right)
\in[0,1]^{B\times3\times3H\times3W}.
$$

$T_{\theta,4}$ is fully fine-tuned Swin2SR at its native x4 scale and $R_{4\rightarrow3}$ is the declared antialiased comparison resize. Code: `Swin2SRTeacher.forward` and `faithsr.pretrained_finetune`.

All 12,239,283 parameters are trainable. Three AdamW epochs minimize Smooth-L1 against the same native-L2A target protocol.

## Layers and activations

- Shallow convolution embeds RGB evidence.
- Shifted-window SwinV2 attention shares information inside local windows and across shifted windows.
- LayerNorm stabilizes transformer activations.
- GELU supplies smooth MLP non-linearity.
- Residual groups preserve low-frequency content and improve gradient flow.
- The inherited convolution plus PixelShuffle(4) reconstruction head remains unchanged.
- The x4→x3 resize is outside the pretrained architecture and is reported explicitly.

## Result

| Method | PSNR | SSIM | False-edge | LPIPS |
|---|---:|---:|---:|---:|
| Bicubic | 23.41633 | 0.59057 | — | 0.62166 |
| Frozen Swin2SR | 21.92152 | 0.54311 | — | 0.47065 |
| Fine-tuned Swin2SR | **23.77746** | **0.61740** | improved by 0.00821 | **0.55062** |

Fine-tuned minus bicubic PSNR was +0.33646 dB with 95% scene-cluster interval +0.04417 to +0.62876. SSIM, false-edge, SAM and LPIPS also improved. Frozen performance was poor, so the result supports domain personalization, not generic pretraining alone.

## Version and decision

v1 and v2 had figure-label/layout defects but numerically valid training. v3 corrected the evidence panel and is accepted. The model became the fidelity reference, not the final deployment candidate: it is about 104× larger than the custom student and still contains an x4→x3 comparison resize.

Artifacts: [`swin2sr_full_finetune_native_l2a_v3`](../../../../artifacts/training_runs/faithsr3x/swin2sr_full_finetune_native_l2a_v3/).

References: Conde et al., *Swin2SR* (2022); Liu et al., *Swin Transformer V2* (2022).
