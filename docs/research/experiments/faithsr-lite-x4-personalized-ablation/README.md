# FaithSR-Lite-X4 Personalized Ablation

Status: **completed mixed/negative result**. Latency and TIFF integrity passed; the registered quality-contribution rule did not.

Detailed preregistration: [`plan.md`](plan.md). Frozen outputs: [`artifacts`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/).

## Research question and connection

Personalized native-x4 Swin2SR improved fidelity but needed 3.42 seconds for one 512 tile. Its weakest profile was heavy compression. This experiment asked whether a roughly 0.06–0.12M parameter student could:

1. keep inference below one second;
2. inherit the personalized teacher's useful behavior;
3. improve heavy-compression robustness without damaging fidelity.

## Four frozen variants

| Variant | Teacher loss | Degradation FiLM and compression refiner | Observation correction | Parameters |
|---|---|---|---|---:|
| Base | No | No | No | 62,502 |
| Distilled | Yes | No | No | 62,502 |
| Compression | No | Yes | Yes | 121,469 |
| Full | Yes | Yes | Yes | 121,469 |

All other data, seed, epochs, optimizer and evaluation settings are identical.

## Mathematical forward function

The observed input and reconstructed output are

$$
\mathbf{x}\in[0,1]^{B\times3\times H\times W},
\qquad
\hat{\mathbf{y}}\in[0,1]^{B\times3\times4H\times4W}.
$$

Here $B$ is batch size, the second dimension contains RGB bands, and $H,W$ are the input height and width. Shallow features are

$$
\mathbf{f}_0=C_{3\times3}(\mathbf{x}).
$$

Each of the six blocks first performs channel mixing, then spatial filtering:

$$
\mathbf{r}_i=P_{2,i}\!\left(\operatorname{GELU}\!\left(D_{7\times7,i}\!\left(\operatorname{GELU}\!\left(P_{1,i}(\mathbf{f}_{i-1})\right)\right)\right)\right),
$$

$$
\mathbf{f}_i=\mathbf{f}_{i-1}+s_i
\left[(1+\boldsymbol{\gamma}_i(\mathbf{z}))\odot\mathbf{r}_i+\boldsymbol{\beta}_i(\mathbf{z})\right]
\odot\sigma\!\left(G_i(\mathbf{r}_i)\right).
$$

$P_1,P_2$ are $1\times1$ pointwise convolutions, $D_{7\times7}$ is a depthwise convolution, $s_i$ is a learned residual scale, $\mathbf{z}$ is the degradation embedding, and $\gamma,\beta$ are FiLM scale and shift. Base/Distilled disable degradation FiLM, so their blocks reduce to an unguided residual form.

Features from all blocks are concatenated and fused:

$$
\mathbf{f}_{F}=C_{3\times3}\!\left(\operatorname{GELU}\!\left(C_{1\times1}[\mathbf{f}_1,\ldots,\mathbf{f}_6]\right)\right)+\mathbf{f}_0.
$$

For Compression/Full, predicted compression $\hat c\in[0,1]$ gates a depthwise refinement branch:

$$
\mathbf{f}_{C}=\mathbf{f}_{F}+s_C\hat c\,\sigma(W_C\mathbf{z})\odot
P_C\!\left(\operatorname{GELU}(D_C(\mathbf{f}_{F}))\right).
$$

Native x4 reconstruction is

$$
\hat{\mathbf{y}}_0=B_4(\mathbf{x})+operatorname{PixelShuffle}_4(C_R(\mathbf{f}_C)),
$$

where $B_4$ is bicubic x4 interpolation and $C_R$ produces $3\times4^2=48$ channels before PixelShuffle. Observation-enabled variants predict a positive unit-sum kernel $\mathbf{k}_{\mathbf{z}}=\operatorname{Softmax}(W_k\mathbf{z})$, re-observe the tentative output, and add a bounded correction:

$$
\tilde{\mathbf{x}}=\downarrow_4(\mathbf{k}_{\mathbf{z}}*\hat{\mathbf{y}}_0),
$$

$$
\hat{\mathbf{y}}=\operatorname{clip}\!\left(
\hat{\mathbf{y}}_0+\alpha(\mathbf{z})B_4(\mathbf{x}-\tilde{\mathbf{x}}),0,1\right),
\qquad 0\leq\alpha\leq0.20.
$$

Why these layers:

- **1×1 pointwise convolution** mixes channel information cheaply.
- **7×7 depthwise convolution** increases local spatial context without a full dense 7×7 cost.
- **GELU** provides smooth non-linear gating and stable residual learning.
- **FiLM** makes feature channels respond to estimated degradation.
- **Sigmoid gates** constrain compression strength and correction magnitude between interpretable bounds.
- **PixelShuffle(4)** gives exact native x4 output while most compute remains on the LR grid.
- **No uncertainty head:** the previous native-L2A calibration was negative.

### Code-to-equation mapping

| Mathematical component | Code implementation | Purpose |
|---|---|---|
| Complete $F_{\theta}(\mathbf{x})$ | `FaithSRLiteX4.forward` in `src/faithsr/lite_x4.py` | Executes feature extraction, fusion, reconstruction and optional correction |
| $\mathbf{z},\hat c$ | `DegradationEncoder` in `src/faithsr/model.py` | Predicts input-conditioned embedding and degradation values |
| $\mathbf{f}_i$ | `AdaptiveBlock` in `src/faithsr/model.py` | Pointwise/depthwise residual processing with optional FiLM |
| $\mathbf{f}_C$ | `CompressionRefinementBlock.forward` | Compression-gated LR feature correction |
| $\tilde{\mathbf{x}}$ | `LearnedObservationOperator.forward` | Projects SR output back onto the observed LR grid |
| teacher target | `Swin2SRTeacher.forward` in `src/faithsr/teacher.py` | Training-only personalized teacher; absent from application inference |
| $\mathcal{L}$ | `CompressionAwareObjective.forward` in `src/faithsr/losses.py` | Computes the registered training losses |

### Layer specification for Full

| Module | Exact design | Parameters |
|---|---|---:|
| Degradation encoder | RGB convolutional encoder to 48-value embedding and four bounded severities | 34,004 |
| Shallow stem | $3\rightarrow32$ channels, $3\times3$, stride 1, padding 1 | 896 |
| Six adaptive blocks | $1\times1$ pointwise, GELU, $7\times7$ depthwise, GELU, $1\times1$ pointwise, FiLM and residual gate | 51,126 |
| Multi-depth fusion | Concatenate six 32-channel outputs, $192\rightarrow32$ with $1\times1$, GELU, then $3\times3$ | 15,424 |
| Compression refiner | $3\times3$ depthwise, GELU, $1\times1$ pointwise and degradation-conditioned sigmoid gate | 2,945 |
| Reconstruction | $32\rightarrow48$ channels using $3\times3$, then PixelShuffle(4) to RGB | 13,872 |
| Observation operator | embedding-conditioned positive $7\times7$ kernel and x4 projection | 2,401 |
| Consistency gate | $48\rightarrow16\rightarrow1$ MLP, GELU then Sigmoid | 801 |

Full contains 121,469 trainable parameters. Base/Distilled remove the encoder, compression refiner, observation operator and consistency gate, leaving 62,502 parameters.

### Activation functions

GELU is

$$
\operatorname{GELU}(u)=u\,\Phi(u),
$$

where $\Phi$ is the standard normal cumulative distribution. It is used after feature projections because its smooth response retains small negative evidence and gives stable residual optimization.

Sigmoid is

$$
\sigma(u)=\frac{1}{1+e^{-u}},\qquad \sigma(u)\in(0,1).
$$

It bounds predicted degradation severities, feature gates and correction strength. Softmax is

$$
\operatorname{Softmax}(u_i)=\frac{e^{u_i}}{\sum_j e^{u_j}},
$$

and constrains the learned observation-kernel coefficients to be positive and sum to one. PixelShuffle is a deterministic channel-to-space rearrangement, not an activation.

## Training loss

All variants use robust Charbonnier reconstruction plus SSIM, gradient and frequency terms. Compression variants add re-observation and degradation-prediction terms. Distilled and Full add teacher pixel and teacher edge terms.

The complete registered objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{Char},c}
+0.10\mathcal{L}_{\mathrm{SSIM}}
+0.03\mathcal{L}_{\nabla}
+0.02\mathcal{L}_{\mathrm{FFT}}
+0.05\mathcal{L}_{\mathrm{obs}}
+0.02\mathcal{L}_{\mathrm{deg}}
+0.05\mathcal{L}_{\mathrm{teacher}}
+0.01\mathcal{L}_{\mathrm{teacher\text{-}edge}}.
$$

$\mathcal{L}_{\mathrm{Char},c}$ is compression-weighted Charbonnier reconstruction; $\mathcal{L}_{\mathrm{SSIM}}$ preserves local structure; $\mathcal{L}_{\nabla}$ compares Sobel edge magnitude; $\mathcal{L}_{\mathrm{FFT}}$ compares Fourier magnitude; $\mathcal{L}_{\mathrm{obs}}$ penalizes disagreement with the LR observation; $\mathcal{L}_{\mathrm{deg}}$ supervises known synthetic degradation values; and the two teacher terms compare output pixels and edges. Terms belonging to disabled modules are set to zero.

Compression samples receive larger reconstruction weight. The teacher is frozen and only the student is updated.

## Training and evaluation protocol

- Same native Sentinel-2 L2A corpus and complete-scene seed-0 split as the accepted v6 experiment: 39 train and 47 held-out tiles.
- Scale x4; 60 epochs; batch 8; 192×192 HR crops; AdamW; cosine learning rate; AMP.
- Personalized teacher checkpoint SHA-256: `b6ee155e167253fd9db8335338fffba342e2af1a201a1662942aeaa1c5949c7a`.
- Seven profiles × 47 images = 329 frozen paired cases per variant.
- Kaggle Tesla P100, PyTorch 2.5.1 CUDA 12.1.
- Best checkpoint: maximum validation PSNR.

## Main results

| Variant | PSNR | SSIM | False-edge | SAM | ERGAS | LPIPS | Observation L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | **23.68809** | **0.61846** | 0.12347 | 0.37636 | **19.27263** | 0.55761 | **0.01748** |
| Distilled | **23.68923** | 0.61839 | 0.12326 | 0.37625 | **19.27183** | 0.55893 | **0.01747** |
| Compression | 23.64276 | 0.61754 | 0.12242 | 0.37612 | 19.35876 | **0.54305** | 0.01804 |
| Full | 23.64512 | 0.61761 | **0.12220** | **0.37599** | 19.35022 | 0.54510 | 0.01800 |

Lower is better for false-edge, SAM, ERGAS, LPIPS and observation L1.

Versus bicubic:

- Base PSNR: −0.00365 dB; interval −0.03594 to +0.02865 (inconclusive).
- Distilled PSNR: −0.00261 dB; interval −0.03446 to +0.02925 (inconclusive).
- Compression PSNR: −0.04650 dB; interval −0.07927 to −0.01374 (worse).
- Full PSNR: −0.04427 dB; interval −0.07669 to −0.01185 (worse).
- All variants improved SSIM, LPIPS and LR re-observation versus bicubic, but none dominated the fidelity metrics.

## What personalization and compression actually contributed

Matched **Distilled minus Base**:

- PSNR +0.00104 dB; interval +0.00059 to +0.00148.
- False-edge −0.00018, but SSIM −0.000068 and LPIPS **worsened** by +0.00128.
- Effect size is too small and mixed for a meaningful distillation claim.

Matched **Compression minus Base**:

- Overall PSNR −0.04286 dB; interval −0.05702 to −0.02869.
- LPIPS improved by −0.01565.
- Heavy-compression PSNR improved by +0.01471 dB, but its interval −0.00039 to +0.02980 crosses zero.
- Overall observation L1 worsened by +0.000595. The preregistered compression rule therefore failed.

Matched **Full minus Compression**:

- PSNR +0.00223 dB and false-edge −0.00019, both tiny.
- LPIPS worsened by +0.00208.

Full minus Base improved heavy-compression PSNR by +0.01547 dB with interval +0.00088 to +0.03007, but overall PSNR fell 0.04063 dB and observation L1 worsened. This is a trade-off, not a win.

## Latency and application pipeline

| Variant | FP32 model P95 | FP16 model P95 | Result |
|---|---:|---:|---|
| Base | **23.35 ms** | **22.39 ms** | Pass |
| Distilled | 23.44 ms | 22.49 ms | Pass |
| Compression | 73.51 ms | 72.85 ms | Pass |
| Full | 73.35 ms | 72.78 ms | Pass |

Full end-to-end TIFF P95 was 482.85 ms FP32 and 399.19 ms FP16 for 512→2048 uncompressed output. Six audits preserved uint8, EPSG:32643, seven tags, RGB descriptions, mask and x4 transform; warnings were empty.

## Visual audit

Worst, median and best stress cases plus one training preview were inspected. Outputs are coherent and no obvious invented road/building geometry was observed, but they are soft and very close to bicubic. The worst Full case was 19.83 dB versus bicubic 20.41 dB. Visual plausibility does not override the negative PSNR evidence.

Saved views: [`Full TIFF pipeline preview`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/pipeline-fp32/pipeline_0-preview.png), [`Full worst stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/worst-native-x4.png), [`Full median stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/median-native-x4.png), and [`Full best stress case`](../../../../artifacts/training_runs/faithsr3x/faithsr_lite_x4_ablation_v1/evaluation/full/stress/best-native-x4.png). Each stress PNG has a same-name JSON evidence record.

## Final decision and next direction

- **Do not promote Full as the final model.** It meets speed/format requirements but not the registered quality rule.
- **Do not claim successful distillation.** The teacher effect is statistically detectable only because rows are highly paired; it is not practically meaningful and worsens LPIPS.
- **Do not add Mamba yet.** No evidence identifies global context as the bottleneck.
- Keep Base as the lightweight latency floor and personalized Swin2SR as the quality ceiling.
- Next experiment should improve capacity/training supervision between these two points, using training-only Sentinel evidence and an untouched external test. Candidate changes must be individually ablated.

## Reproducibility and references

Kernel: `muhammadhassan2/faithsr-lite-x4-personalized-ablation`, private version 1. Execution source commit: `3556eb0`. Correct source dataset: private version 19; version 18 was an incomplete directory upload and was not used.

References: Conde et al., *Swin2SR* (2022); Shi et al., *PixelShuffle* (2016); Perez et al., *FiLM* (2017); Hendrycks and Gimpel, *GELU* (2016); Wang et al., *Real-ESRGAN* (2021); Wang et al., *SSIM* (2004).
