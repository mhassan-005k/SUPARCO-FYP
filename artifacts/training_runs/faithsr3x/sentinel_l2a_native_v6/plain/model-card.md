# Experiment Model Card: plain

## Purpose and lineage

- Why this experiment was run: Establish a native-L2A custom-backbone control under scene-separated evaluation.
- Previous experiment/connection: Delhi RGB ablations used display tiles without native L2A radiometric provenance.
- Scale: exact native 3x
- Trainable parameters: 56,433
- Degradation adaptation: False
- Learned observation consistency: False
- Spatial uncertainty: False

## Mathematical forward function

\[
f_0=C_{3\times3}(x),\qquad
h_i=P_{2,i}(\operatorname{GELU}(D_i(\operatorname{GELU}(P_{1,i}(f_{i-1}))))),
\]

\[
f_i=f_{i-1}+s_i h_i\odot\sigma(G_i(h_i)),\qquad
\hat y_0=\operatorname{Bicubic}_3(x)+\operatorname{PixelShuffle}_3(C_r(f)).
\]

This control variant intentionally omits degradation conditioning, learned observation correction, and uncertainty.

The training objective and degradation equations are frozen in `docs/research/model-and-experiment-methodology.md` and instantiated by the exact configuration below.

## Architecture figure

```mermaid
flowchart LR
    X["LR input"] --> S["3x3 stem"]
    X --> E["Degradation encoder: False"]
    S --> B["Six residual blocks; FiLM=False"]
    E --> B
    B --> F["Multi-depth fusion"]
    F --> P["Conv + PixelShuffle x3"]
    X --> C["Bicubic base"]
    P --> Y0["Initial SR"]
    C --> Y0
    Y0 --> O["Learned observation correction=False"]
    O --> Y["Final SR"]
    F --> U["Uncertainty head=False"]
```

## Top-level layers and parameters

| Name | Layer/module | Parameters |
|---|---|---:|
| `shallow` | Conv2d | 896 |
| `blocks` | ModuleList | 32,310 |
| `fusion` | Sequential | 15,424 |
| `reconstruction` | Sequential | 7,803 |

## Activations and rationale

| Activation | Module count | Why used |
|---|---:|---|
| GELU | 19 | Smooth feature transformation; preserves small negative evidence. |
| Sigmoid | 6 | Bounds gates, severities, or uncertainty to [0,1]. |

PixelShuffle is a deterministic tensor rearrangement, not an activation. For RGB x3, the preceding convolution predicts 3x3²=27 sub-pixel channels.

## Exact configuration

```yaml
seed: 0
scale: 3
model:
  channels: 32
  num_blocks: 6
  embed_dim: 48
  kernel_size: 7
  use_degradation_adaptation: false
  use_observation_consistency: false
  use_uncertainty: false
data:
  crop_size: 192
  train_ratio: 0.5
  split_grid_degrees: 0.1
  extensions:
  - .tif
  - .tiff
  mixed_probability: 0.75
degradation:
  curriculum_epochs: 15
  start_severity: 0.2
  mtf_probability: 0.5
training:
  epochs: 60
  batch_size: 16
  validation_batch_size: 16
  workers: 2
  learning_rate: 0.0002
  weight_decay: 0.0001
  amp: true
loss:
  charbonnier: 1.0
  ssim: 0.1
  gradient: 0.03
  frequency: 0.02
  observation: 0.05
  degradation: 0.02
  uncertainty: 0.02
  distillation: 0.0
inference:
  tile_size: 512
  overlap: 32
```

## References

- Swin2SR: https://arxiv.org/abs/2209.11345
- GELU: https://arxiv.org/abs/1606.08415
- FiLM: https://arxiv.org/abs/1709.07871
- PixelShuffle/sub-pixel convolution: https://arxiv.org/abs/1609.05158
- Canonical project methodology: `docs/research/model-and-experiment-methodology.md`


<!-- GENERATED_RESULTS_START -->
## Measured training result

Checkpoint selection rule: maximum validation PSNR. These values come directly from `metrics.csv`.
Training duration: 116.39 seconds.

| Checkpoint row | Epoch | train_loss | psnr | ssim | false_edge_rate |
|---|---:|---:|---:|---:|---:|
| Best | 58 | 0.09339773903290431 | 22.52643330891927 | 0.5905364751815796 | 0.13298436999320984 |
| Final | 60 | 0.09509493907292683 | 22.526416778564453 | 0.5905160705248514 | 0.1329933007558187 |
<!-- GENERATED_RESULTS_END -->
