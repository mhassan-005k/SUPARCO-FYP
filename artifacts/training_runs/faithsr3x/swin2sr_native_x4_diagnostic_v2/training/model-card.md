# Model Card: Fully Fine-Tuned Swin2SR Native x4

## Role and necessity

This is a strong personalized pretrained candidate, not a renamed custom FaithSR architecture. Every parameter of pinned pretrained `caidas/swin2SR-classical-sr-x4-64` at revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e` is trainable. Target scale is x4; native x4 output is supervised directly without a resize.

## Mathematical function

For LR input `x`, the evaluated mapping is:

`y_hat = Swin2SR_x4(x)`.

All 12,239,283 parameters minimize `SmoothL1(y_hat, y)` for 3 epochs with AdamW learning rate 1e-05. The resize is not hidden and this baseline is never called native x3.

## Layers and activations

Swin2SR uses shallow convolutional embedding, residual Swin Transformer groups with shifted-window self-attention, GELU MLPs, layer normalization, convolutional residual connections, and an x4 sub-pixel reconstruction head. GELU is inherited from the published pretrained architecture because its smooth gating is standard inside transformer MLPs. PixelShuffle is inherited for x4 reconstruction. Any x4-to-x3 resize belongs only to the earlier matched-x3 comparison protocol, not the minimum-3x product requirement.

```mermaid
flowchart LR
    LR["Synthetic LR"] --> EMB["Shallow convolution"]
    EMB --> RSTB["Residual Swin Transformer groups
window attention + GELU MLP"]
    RSTB --> X4["Pretrained x4 reconstruction"]
    X4 --> LOSS["Smooth-L1 against native-x4 held-out HR"]
```

## Interpretation rule

Frozen and fully fine-tuned results use the same scene-separated validation files and degradation records. A visually pleasing example cannot replace the saved per-image table, paired scene-cluster intervals, or negative failure cases.

## References

- Conde et al., “Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration,” ECCV Workshops 2022: https://arxiv.org/abs/2209.11345
- Immutable model configuration used here: https://huggingface.co/caidas/swin2SR-classical-sr-x4-64/blob/c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e/config.json
- Project architecture and experiment rationale: `docs/research/model-and-experiment-methodology.md`
