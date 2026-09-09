# Model Card: Pinned Pretrained Swin2SR x4 Reference

## Role and experiment connection

This is a baseline/frozen teacher, not the proposed architecture. It was needed to prevent comparison only against interpolation and to transfer a pretrained restoration prior into the custom FaithSR student. Model ID: `caidas/swin2SR-classical-sr-x4-64`; immutable revision: `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`; Apache-2.0; about 12.2M parameters.

## Function, layers, and activations

For window features (X), shifted-window attention uses

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt d+B)V,
\]

followed by residual LayerNorm/MLP paths. The pinned config uses embed dimension 180, six residual Swin stages, depth `[6,6,6,6,6,6]`, six heads per stage, window 8, MLP ratio 2, GELU, and PixelShuffle x4. For this fixed-x3 project:

\[
t_3(x)=\mathrm{AntialiasedBicubicResize}_{3H,3W}(f_{Swin2SR,x4}(x)).
\]

| Component | Configuration | Why / limitation |
|---|---|---|
| Shallow convolution | RGB to 180-dimensional features | Local evidence before transformer processing. |
| 6 residual Swin stages | 6 blocks each, 6 heads, window 8 | Shifted local attention captures wider context efficiently. |
| MLP activation | GELU, ratio 2 | Smooth nonlinear token/channel transform. |
| Attention activation | softmax | Normalized evidence weighting within windows. |
| Reconstruction | PixelShuffle x4 | Checkpoint-native scale; it is not native x3. |
| Adaptation | antialiased resize x4→x3 | Makes dimensions comparable but remains a baseline limitation. |

```mermaid
flowchart LR
    X["LR RGB"] --> C["Shallow conv, dim 180"] --> S["6x6 SwinV2 blocks"]
    S --> P["PixelShuffle x4"] --> R["Explicit antialiased resize to x3"] --> T["Baseline / teacher target"]
```

Why activations: attention softmax gives normalized within-window weights; GELU provides smooth MLP nonlinearity; PixelShuffle rearranges predicted subpixels and is not an activation.

References: [pinned config](https://huggingface.co/caidas/swin2SR-classical-sr-x4-64/blob/c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e/config.json), [Swin2SR paper](https://arxiv.org/abs/2209.11345), and [`model-and-experiment-methodology.md`](../model-and-experiment-methodology.md).
