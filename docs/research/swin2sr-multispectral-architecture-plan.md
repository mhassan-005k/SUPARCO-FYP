# Multispectral Radiometric Swin2SR Architecture Plan

Status: Selected successor to the closed lightweight hybrid branch

Plan date: 2026-09-19

## Research objective

The next model will test one narrow architectural hypothesis: a pretrained Swin2SR spatial trunk can be personalized for multispectral satellite imagery by adding a lightweight spectral adapter and a band-aware reconstruction head, while preserving radiometric values, geometric alignment, geospatial metadata, and the measured five-second latency requirement.

The working name is **MSR Swin2SR**, meaning multispectral and radiometric Swin2SR. The name is provisional and does not imply novelty until the literature gap and ablation results support it.

## Baseline retained without change

The baseline is the pinned native-x4 checkpoint `caidas/swin2SR-classical-sr-x4-64` at revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`.

The retained trunk contains:

- a shallow 3 by 3 convolution;
- six residual SwinV2 Transformer groups;
- six shifted-window SwinV2 layers per group;
- 180 feature channels;
- window size 8;
- six attention heads;
- LayerNorm;
- scaled cosine window attention with relative position bias;
- GELU multilayer perceptrons;
- residual convolutions and skip connections;
- a native x4 PixelShuffle reconstruction path.

The completed personalized baseline has 12,239,283 trainable parameters and achieved a scene-cluster PSNR gain of 0.50377 dB and an SSIM gain of 0.04228 over bicubic across 329 stress cases. Its FP32 model-only P95 latency was 3.419 seconds and its full TIFF P95 was 3.791 seconds on the recorded Tesla P100, both within the SRS v1.2 five-second limit.

## Proposed architectural change

### Multispectral input adapter

For input with `C` declared spectral bands,

$$
\mathbf{x}\in[0,1]^{B\times C\times H\times W},
$$

each band first receives a depthwise 3 by 3 spatial projection:

$$
\mathbf{b}=\operatorname{DWConv}_{3\times3}(\mathbf{x}).
$$

A 1 by 1 cross-band projection maps the declared bands into the pretrained 180-channel feature space:

$$
\mathbf{f}_0=\operatorname{Conv}_{1\times1}^{C\rightarrow180}(\mathbf{b}).
$$

This separates within-band spatial evidence from cross-band mixing. The first registered multispectral profile uses Sentinel-2 B2, B3, B4, and B8 on their shared native 10 m grid. Full 13-band processing is excluded until the 10 m, 20 m, and 60 m alignment policy is implemented and validated.

### Spectral channel gate

Global average pooling summarizes each feature channel:

$$
\mathbf{q}=\operatorname{GAP}(\mathbf{f}_0).
$$

The spectral gate is:

$$
\mathbf{a}=\sigma\left(\mathbf{W}_2\operatorname{GELU}(\mathbf{W}_1\mathbf{q})\right),
\qquad
\mathbf{a}\in(0,1)^{B\times180}.
$$

The gated shallow features are:

$$
\mathbf{f}_{\mathrm{in}}=\mathbf{f}_0+\mathbf{a}\odot\mathbf{f}_0.
$$

The reduction ratio and exact parameter count will be frozen before training. This gate is the only new feature-selection module in the first ablation. No context cascade, Mamba block, degradation encoder, teacher-student path, or uncertainty head will be added.

### Pretrained spatial trunk

The adapted input passes through the unchanged Swin2SR trunk:

$$
\mathbf{f}_{\mathrm{deep}}=T_{\theta}(\mathbf{f}_{\mathrm{in}}),
$$

where `T` contains the residual SwinV2 groups. For each window, scaled cosine attention is:

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=
\operatorname{Softmax}\left(\frac{\cos(\mathbf{Q},\mathbf{K})}{\tau}+\mathbf{S}\right)\mathbf{V}.
$$

Here, `tau` is the learned per-head temperature and `S` is the continuous relative position bias. Shifted windows permit cross-window interaction without full-image quadratic attention.

### Band-aware reconstruction head

For native scale `r=4`, the output head predicts `C times r squared` channels:

$$
\widehat{\mathbf{y}}=operatorname{Bicubic}_{4}(\mathbf{x})+
\operatorname{PixelShuffle}_{4}\left(
\operatorname{Conv}_{3\times3}^{180\rightarrow16C}(\mathbf{f}_{\mathrm{deep}})
\right).
$$

For the four-band profile, the final convolution therefore predicts 64 channels and PixelShuffle rearranges them into four output bands at `4H by 4W`. No RGB-only output will be used to silently reattach an untouched NIR band.

## Activation and normalization functions

| Function | Formula or operation | Location | Reason |
|---|---|---|---|
| GELU | `GELU(t) = t Phi(t)` | Spectral gate and inherited transformer MLPs | Preserves smooth positive and negative feature evidence and matches pretrained trunk behavior |
| Sigmoid | `sigma(t) = 1 / (1 + exp(-t))` | Spectral channel gate | Bounds multiplicative channel weights from zero to one |
| Softmax | `exp(t_i) / sum_j exp(t_j)` | Window attention | Produces normalized attention weights |
| LayerNorm | channelwise normalized affine transform | Every inherited SwinV2 layer | Stabilizes deep transformer optimization |
| PixelShuffle | deterministic channel-to-space rearrangement | Reconstruction head | Produces exact native spatial scaling without a learned post-output resize |

PixelShuffle and LayerNorm are not activation functions. Their roles will remain separate in the paper and implementation documentation.

## Radiometric input and output contract

The implementation treats 8-bit, 10-bit, and 12-bit inputs as separate declared radiometric profiles rather than silently converting every file to an 8-bit display image.

Bit depth is handled at the raster boundary rather than inferred by the network. For declared effective bit depth `b`:

$$
DN_{\max}=2^b-1,
\qquad
\mathbf{x}=\frac{\operatorname{clip}(DN,0,DN_{\max})}{DN_{\max}},
$$

$$
\widehat{DN}=\operatorname{round}\left(
DN_{\max}\operatorname{clip}(\widehat{\mathbf{y}},0,1)
\right).
$$

Ten-bit and 12-bit products remain in a compatible integer container such as uint16, while their effective bit depth, no-data value, scale, offset, units, band descriptions, CRS, affine transform, and masks are preserved. Ambiguous uint16 input will be rejected unless a product profile or authoritative tag declares its effective depth.

## Training objective

The first multispectral experiment uses a fidelity objective rather than an adversarial objective:

$$
\mathcal{L}=\lambda_c\mathcal{L}_{\mathrm{Char}}
+\lambda_s\mathcal{L}_{\mathrm{SSIM}}
+\lambda_r\mathcal{L}_{\mathrm{RMSE}}
+\lambda_{sam}\mathcal{L}_{\mathrm{SAM}}
+\lambda_g\mathcal{L}_{\mathrm{grad}}
+\lambda_o\mathcal{L}_{\mathrm{obs}}.
$$

The terms are:

$$
\mathcal{L}_{\mathrm{Char}}=
\frac{1}{N}\sum_i\sqrt{(\widehat y_i-y_i)^2+\epsilon^2},
$$

$$
\mathcal{L}_{\mathrm{RMSE}}=
\sqrt{\frac{1}{N}\sum_i(\widehat y_i-y_i)^2},
$$

$$
\mathcal{L}_{\mathrm{SAM}}=
\frac{1}{P}\sum_p
\arccos\left(
\frac{\widehat{\mathbf y}_p^T\mathbf y_p}
{\|\widehat{\mathbf y}_p\|_2\|\mathbf y_p\|_2+\epsilon}
\right),
$$

$$
\mathcal{L}_{\mathrm{obs}}=
\left\|D(\widehat{\mathbf y})-\mathbf x\right\|_1.
$$

Loss weights will be pre-registered. An additional loss is retained only if a matched ablation improves the targeted metric without harming false-edge rate, spectral angle, registration, or metadata integrity.

## Registration policy

Training pairs must be co-registered before cropping. Global phase correlation provides an initial translation estimate; local feature or control-point displacement verifies difficult scenes. A pair is rejected or corrected before training when its registered error exceeds the frozen data threshold.

For `M` matched displacements:

$$
e_{reg}=\frac{1}{M}\sum_{i=1}^{M}\sqrt{\Delta x_i^2+\Delta y_i^2}.
$$

Final SRS compliance requires `e_reg <= 0.5` output pixels. The evaluation must not hide registration error by choosing a favourable post-hoc shift independently for each model.

## Scale policy

The first implementation remains native x4 because it reuses the strongest verified checkpoint. A later scale-conditioned reconstruction stage may provide separate x2 and x4 heads over the shared trunk. An x8 head will be considered only when registered real-pair data support it. Cascading an x4 model or resizing its output will not be described as newly recovered physical resolution.

For a desired 0.3 m grid:

$$
s_{requested}=\frac{GSD_{input}}{0.3}.
$$

Inputs at 0.5 m, 1 m, and 2.5 m therefore require nominal ratios of approximately 1.67, 3.33, and 8.33. These ratios define output sampling grids, not guaranteed sensor-equivalent information.

## Registered experiment matrix

Only two new architecture variants are permitted before a continuation decision:

| Variant | Input | Change from baseline | Purpose |
|---|---|---|---|
| S0 Personalized Swin2SR | RGB | No architecture change | Reproduce the strongest accepted baseline with RMSE and registration added |
| S1 Multispectral Adapter | B2 B3 B4 B8 | New input adapter, spectral gate, four-band head | Test whether explicit cross-band processing improves spatial and spectral fidelity |
| S2 Registered Fidelity | B2 B3 B4 B8 | S1 plus pre-registered RMSE, SAM, gradient, and observation terms | Test the training objective without another architectural module |

## Data and evaluation

- EuroSAT multispectral TIFFs may be used for representation pretraining and land-cover diversity, but not as real x4 ground truth.
- Native Sentinel-2 L2A scenes remain the geospatial, 512-pixel stress, and out-of-distribution source.
- OpenSR-test, SEN2VENUS, WorldStrat, or permitted SUPARCO references provide real cross-sensor evaluation after registration and product-level audit.
- Splits are separated by scene and geography.
- The independent final set is frozen before selecting S1 or S2.

Metrics are PSNR, SSIM, RMSE reduction, mean registration error, SAM, ERGAS, false-edge rate, LPIPS, observation L1, latency, VRAM, and metadata preservation.

## Stop rules

The spectral adapter is rejected if it does not improve the personalized Swin2SR baseline with a paired confidence interval above zero for its primary reconstruction metric, or if it materially worsens SAM, ERGAS, false-edge rate, registration, or metadata preservation.

The successor track stops architecture development after S2 unless a documented failure identifies one specific missing mechanism. No new block is added because it is fashionable or because another paper used it.

Final deployment requires:

- PSNR gain at least 1.5 dB over bicubic or sufficient total SRS score under the frozen acceptance rule;
- SSIM at least 0.90 or gain at least 0.03;
- RMSE reduction at least 10 percent;
- registration error at most 0.5 output pixels;
- 100 percent tested geospatial metadata preservation;
- 512 by 512 processing within five seconds on the declared target GPU;
- no visually significant checkerboard, ringing, or misleading texture.
