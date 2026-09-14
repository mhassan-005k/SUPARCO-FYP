# Experiment Model Card: faithsr-lite-x4-full

## Purpose and lineage

- Why this experiment was run: Test whether personalized-teacher distillation and the diagnosed compression mechanism are complementary.
- Previous experiment/connection: swin2sr-native-x4-diagnostic-v2
- Scale: exact native 4x
- Trainable parameters: 121,469
- Degradation adaptation: True
- Learned observation consistency: True
- Spatial uncertainty: False
- Compression-gated refinement: True

## Mathematical forward function

\[
f_0=C_{3\times3}(x),\qquad
h_i=P_{2,i}(\operatorname{GELU}(D_i(\operatorname{GELU}(P_{1,i}(f_{i-1}))))),
\]

\[
f_i=f_{i-1}+s_i h_i\odot\sigma(G_i(h_i)),\qquad
\hat y_0=\operatorname{Bicubic}_4(x)+\operatorname{PixelShuffle}_4(C_r(f)).
\]

\(\bar h_i=(1+\gamma_i(z))\odot h_i+\beta_i(z)\) (degradation-conditioned FiLM).

\(k_z=\operatorname{softmax}(W_kz),\;\tilde x=\downarrow_4(k_z*\hat y_0),\;\hat y=\hat y_0+\alpha(z)\operatorname{Bicubic}_4(x-\tilde x),\;0\le\alpha\le0.25\).

\(f_c=f+s_c\,c(z)\,\sigma(W_cz)\odot R_c(f)\) (compression-gated LR-space refinement).

The training objective and degradation equations are frozen in `docs/research/model-and-experiment-methodology.md` and instantiated by the exact configuration below.

## Architecture figure

```mermaid
flowchart LR
    X["LR input"] --> S["3x3 stem"]
    X --> E["Degradation encoder: True"]
    S --> B["6 residual blocks; FiLM=True"]
    E --> B
    B --> F["Multi-depth fusion"]
    F --> P["Conv + PixelShuffle x4"]
    X --> C["Bicubic base"]
    P --> Y0["Initial SR"]
    C --> Y0
    Y0 --> O["Learned observation correction=True"]
    O --> Y["Final SR"]
    F --> U["Uncertainty head=False"]
```

## Top-level layers and parameters

| Name | Layer/module | Parameters |
|---|---|---:|
| `encoder` | DegradationEncoder | 34,004 |
| `shallow` | Conv2d | 896 |
| `blocks` | ModuleList | 51,126 |
| `fusion` | Sequential | 15,424 |
| `compression_refiner` | CompressionRefinementBlock | 2,945 |
| `reconstruction` | Sequential | 13,872 |
| `observation` | LearnedObservationOperator | 2,401 |
| `consistency_gate` | Sequential | 801 |

## Activations and rationale

| Activation | Module count | Why used |
|---|---:|---|
| GELU | 26 | Smooth feature transformation; preserves small negative evidence. |
| Sigmoid | 9 | Bounds gates, severities, or uncertainty to [0,1]. |

PixelShuffle is a deterministic tensor rearrangement, not an activation. For RGB x4, the preceding convolution predicts 3x4²=48 sub-pixel channels.

## Exact configuration

```yaml
seed: 0
scale: 4
model:
  channels: 32
  num_blocks: 6
  embed_dim: 48
  kernel_size: 7
  max_consistency_strength: 0.2
data:
  crop_size: 192
  train_ratio: 0.5
  split_grid_degrees: 0.1
  extensions:
  - .tif
  - .tiff
  mixed_probability: 0.75
  compression_focus_probability: 0.5
  compression_focus_floor: 0.8
degradation:
  curriculum_epochs: 15
  start_severity: 0.2
  mtf_probability: 0.5
training:
  epochs: 60
  batch_size: 8
  validation_batch_size: 16
  teacher_batch_size: 2
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
  distillation: 0.05
  distillation_edge: 0.01
  compression_focus: 1.0
inference:
  tile_size: 512
  overlap: 32
variant: full
resolved_model:
  in_channels: 3
  out_channels: 3
  channels: 32
  num_blocks: 6
  embed_dim: 48
  kernel_size: 7
  scale: 4
  use_degradation_adaptation: true
  use_observation_consistency: true
  use_compression_refinement: true
  max_consistency_strength: 0.2
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
Training duration: 284.83 seconds.

| Checkpoint row | Epoch | train_loss | psnr | ssim | false_edge_rate | observation_l1 |
|---|---:|---:|---:|---:|---:|---:|
| Best | 59 | 0.10913019627332687 | 21.85740343729655 | 0.52928094069163 | 0.13123214741547903 | 0.022834825639923412 |
| Final | 60 | 0.10805994421243667 | 21.85740343729655 | 0.5292737682660421 | 0.13123203814029694 | 0.02283404270807902 |
<!-- GENERATED_RESULTS_END -->
