# FaithSR Model and Experiment Methodology

Status: Living technical record. Results are claims only where an immutable artifact is linked.

Last updated: 2026-09-09

## Documentation rule for every experiment

Har future experiment ki entry mein yeh cheezen mandatory hain:

1. hypothesis aur experiment karne ki zaroorat;
2. previous experiment se exact connection;
3. dataset/version, split rule, seed aur manifest hash;
4. model ka mathematical forward function;
5. layer-by-layer architecture aur parameter count;
6. activation functions aur unko choose karne ki wajah;
7. loss function with weights;
8. architecture/data-flow figure in Mermaid;
9. hardware, duration, checkpoint-selection rule aur metrics;
10. result, negative result, limitation aur next decision;
11. code/config/checkpoint/result references.

Is template ka maqsad yeh hai ke thesis mein sirf “model train kiya” na likha ho; har design decision traceable aur defendable ho.

The trainer enforces this practice by generating a `model-card.md` inside every new run directory. Researchers pass `--experiment-rationale` and `--previous-experiment`; the card derives active components, layer parameter counts, activations, equations, Mermaid flow, and exact YAML directly from the instantiated model/config. Canonical hand-written interpretation in this document is still required after results arrive.

## Experiment lineage

```mermaid
flowchart LR
    S["Pretrained Swin2SR x4\nqualitative reference"] --> V0["FaithSR v0.1 x4\ncustom lightweight base"]
    V0 --> DA["v0.1 + degradation adaptation"]
    DA --> AOC["v0.1 + fixed bicubic\nobservation correction"]
    AOC --> N3["FaithSR-3X\nnative x3 reconstruction"]
    N3 --> LO["Learned observation operator\n+ bounded consistency"]
    LO --> UQ["Uncertainty head\n+ fidelity metrics"]
    UQ --> E8["EuroSAT v8\npreliminary convergence"]
    E8 --> MC["Matched interpolation comparison"]
    MC --> SD["Sentinel-2 Delhi\ngeographic ablations"]
    SD --> SH["Frozen SUPARCO holdout\nexternal gate"]
```

Roman Urdu explanation: Swin2SR ne strong pretrained reference diya, lekin woh project ki apni architecture nahi thi. Is liye v0.1 lightweight custom model banaya gaya. v0.1 ke small degradation gains aur weaker observation-consistency result ne native 3x model, richer degradation vector aur learned observation operator ki zaroorat establish ki. EuroSAT run ne implementation/convergence verify ki; Sentinel experiment ab geographic generalization aur component attribution test karta hai. SUPARCO holdout last external-validity gate hai.

## Notation

- (x \in [0,1]^{B\times C\times H\times W}): normalized low-resolution input.
- (y \in [0,1]^{B\times C\times 3H\times 3W}): high-resolution target.
- \(\hat y=f_\theta(x)\): model reconstruction.
- (z=E_\phi(x)): degradation embedding.
- (d\): normalized synthetic degradation parameters.
- (D_z(\hat y)): learned, degradation-conditioned observation operator that maps SR back to LR.
- (u\in[0,1]^{B\times1\times3H\times3W}): predicted uncertainty/risk.

## Shared synthetic observation model

The native 3x experiments create an LR observation as

\[
x = Q_8\left(J_q\left(\mathcal{N}_{g,p}\left(\downarrow_3\left(k_{\sigma_x,\sigma_y,\alpha} * y\right)\right)\right)\right),
\]

where (k) is a normalized rotated anisotropic Gaussian kernel, \(\downarrow_3\) is antialiased bicubic or bilinear decimation, \(\mathcal{N}_{g,p}\) combines Gaussian and optional Poisson noise, (J_q) is a JPEG round trip at quality (q\), and (Q_8(v)=\operatorname{round}(255\operatorname{clip}(v,0,1))/255\) is 8-bit quantization.

Why this function: sirf clean bicubic downsampling par training real TIFF/JPEG inputs ke blur, noise, compression aur aliasing ko represent nahi karti. Random parameters exact seed ke sath JSON mein record hote hain, is liye validation degradation replayable hai. Yeh still synthetic model hai; real sensor MTF ka substitute claim nahi kiya jata.

Code: [`src/faithsr/degradation.py`](../../src/faithsr/degradation.py)

## Pretrained Swin2SR reference

### Role

Swin2SR x4 ko do roles diye gaye:

- strong pretrained inference baseline;
- frozen teacher for student distillation.

It is not presented as the proposed model. The pinned baseline is `caidas/swin2SR-classical-sr-x4-64` at revision `c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e`. It contains about 12.2 million parameters and produces x4 output. For the fixed x3 task, its result is antialiased-bicubic resized:

\[
t_3(x)=\mathcal{R}_{3H,3W}\left(f_{\text{Swin2SR-x4}}(x)\right).
\]

Why resize rather than claim native x3: available checkpoint x4 hai. Resize explicitly document karne se comparison reproducible rehta hai aur x4 output ko ghalat taur par native x3 nahi kaha jata. A native/fine-tuned x3 pretrained baseline future stronger comparison hai.

```mermaid
flowchart LR
    LR["LR RGB"] --> SW["Frozen Swin2SR\nwindowed SwinV2 blocks"]
    SW --> X4["PixelShuffle x4"]
    X4 --> RS["Antialiased resize to exact x3"]
    RS --> TB["Teacher/baseline output"]
```

The supplied UC Merced fine-tuned checkpoint is an x4 `Swin2SRForImageSuperResolution` state dictionary. It remains an historical qualitative artifact; it is not silently loaded into a mismatched x3 architecture.

### Supplied qualitative Swin2SR experiment

Why it was run: pretrained transformer ki practical visual capability dekhna aur custom model ke liye strong-reference direction choose karna. Connection: is experiment ne prove nahi kiya ke model project ka apna contribution hai; isi limitation se FaithSR v0.1 start hua.

Existing figures:

- [field LR versus Swin2SR](../../artifacts/results/figures/field_lr_vs_swin2sr.png)
- [field fine-tuning comparison](../../artifacts/results/figures/field_finetune_comparison.png)
- [field method comparison](../../artifacts/results/figures/field_method_comparison.png)
- [airplane method comparison](../../artifacts/results/figures/airplane_method_comparison.png)
- [field real-x4 comparison](../../artifacts/results/figures/field_real_x4_comparison.png)

Interpretation rule: yeh selected qualitative examples architecture feasibility show karte hain, dataset-level superiority nahi. Exact split/metric manifest supplied nahi tha, is liye paper mein in figures ko quantitative evidence ke taur par use nahi karna.

References: [Swin2SR paper](https://arxiv.org/abs/2209.11345), [pinned Hugging Face model](https://huggingface.co/caidas/swin2SR-classical-sr-x4-64/tree/c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e).

## FaithSRNet v0.1 experiments

### Why v0.1 was needed

Pretrained Swin2SR se result mil sakta tha, lekin FYP contribution explainable/custom nahi banti. v0.1 ne lightweight CNN, degradation conditioning aur observation consistency ke hypotheses ko independently test kiya.

### v0.1 forward function

For shallow features (f_0=C_{3\times3}(x)\), six residual blocks compute

\[
r_i = C^{(i)}_{1\times1,\downarrow}\left(\operatorname{GELU}\left(C^{(i)}_{3\times3,dw}\left(\operatorname{GELU}(C^{(i)}_{1\times1,\uparrow}(f_{i-1}))\right)\right)\right),
\]

\[
\tilde r_i=(1+\gamma_i(z))\odot r_i+\beta_i(z),\qquad
f_i=f_{i-1}+s_i\,\tilde r_i\odot\sigma(g_i(\tilde r_i)),
\]

where FiLM terms \(\gamma,\beta\) are absent in the Base variant. Intermediate outputs are fused, PixelShuffle reconstructs a residual, and bicubic interpolation supplies a stable base:

\[
\hat y_0=\operatorname{Bicubic}_s(x)+U_s(F([f_1,\ldots,f_6])+f_0).
\]

The old observation-consistency variant used a fixed bicubic operator:

\[
\hat y=\hat y_0+\alpha(z)\operatorname{Bicubic}_s\left(x-\operatorname{BicubicDown}(\hat y_0)\right).
\]

### v0.1 layers

| Stage | Layer | Purpose and reason |
|---|---|---|
| Degradation encoder | 3 strided 3x3 convolutions: 3→16→32→48, GELU, global average pool, 48→64 linear | Compact global blur/noise/compression context without requiring metadata. |
| Shallow feature | 3x3 convolution, 3→48 | Local spatial evidence ko feature space mein map karta hai. |
| Six residual blocks | 1x1 expansion, 3x3 depthwise, 1x1 projection, FiLM, channel gate | Depthwise separable design compute kam rakhta; residual path training stable rakhta. |
| Fusion | concat six outputs, 1x1 reduction, GELU, 3x3 convolution | Sirf last block par depend karne ke bajaye shallow/deep evidence reuse hota hai. |
| Upsampler | one/two Conv→PixelShuffle(2) stages | v0.1 x2/x4 ke liye efficient sub-pixel reconstruction. |
| Consistency gate | 64→16→1 MLP, GELU then sigmoid | Per-image correction strength ko bounded \([0,1]\) rakhta hai. |

```mermaid
flowchart LR
    X["LR x"] --> DE["Optional degradation encoder E(x)"]
    X --> C0["3x3 shallow conv"]
    DE --> F1["FiLM parameters"]
    C0 --> RB["6 gated depthwise residual blocks"]
    F1 --> RB
    RB --> FU["Multi-block fusion"]
    FU --> UP["x2/x4 PixelShuffle residual"]
    X --> BI["Bicubic base"]
    UP --> ADD["Add"]
    BI --> ADD
    ADD --> OC["Optional fixed bicubic consistency"]
```

### v0.1 experiment connection and result

| Experiment | Parameters | Why it was run | Connection | Complex PSNR / SSIM | Decision |
|---|---:|---|---|---:|---|
| Base x4 | 272,769 | Establish custom lightweight reference without degradation conditioning. | First custom model after Swin2SR reference. | 24.6625 / 0.6502 | Valid base. |
| Base + DA x4 | 334,484 | Test whether input-estimated degradation improves mixed blur/noise/JPEG reconstruction. | Adds encoder + FiLM to Base only. | 24.6759 / 0.6520 | Gain positive but too small for contribution claim. |
| Base + DA + AOC x4 | 335,541 | Test whether output can be pulled back toward observed LR data. | Adds fixed bicubic correction to DA. | 24.6082 / 0.6471 | Regression; fixed operator likely mismatched unknown degradation. Replace it. |

This negative result directly motivated a learned observation operator in FaithSR-3X. Source: [`src/faithsr_v01/faithsrnet_v01.py`](../../src/faithsr_v01/faithsrnet_v01.py); immutable results: [`experiment-log.md`](experiment-log.md).

Per-experiment cards: [Base x4](model-cards/faithsr-v01-base-x4.md), [Degradation-Adaptive x4](model-cards/faithsr-v01-da-x4.md), and [DA + Observation x4](model-cards/faithsr-v01-full-x4.md). The external teacher/baseline has a separate [pinned Swin2SR card](model-cards/swin2sr-pretrained-reference.md).

## FaithSR-3X architecture

### Why a new architecture was required

The current custom output is native x3 and therefore meets the confirmed minimum-x3 requirement, whereas v0.1 was x2/x4. The v0.1 degradation target lacked aliasing, its observation operator was fixed bicubic, and it had no spatial uncertainty. FaithSR-3X addresses those specific gaps while remaining much smaller than Swin2SR; native-x4 pretrained output is also a valid product candidate.

### Layer-by-layer definition (EuroSAT/Sentinel configuration)

| Stage | Exact layer sequence | Output/purpose | Why this design |
|---|---|---|---|
| Degradation encoder | Conv3x3/s2 3→24 → GELU → Conv3x3/s2 24→40 → GELU → Conv3x3/s2 40→56 → GELU → adaptive average pool → Linear 56→48 → GELU | Embedding \(z\in\mathbb{R}^{48}\); auxiliary 4-vector via 48→32→4 sigmoid | Three strided layers cheaply enlarge receptive field; global pool accepts arbitrary tile size; four targets represent blur, noise, compression, aliasing. |
| Shallow stem | Conv3x3 3→32 | \(f_0\) | Preserves full LR grid and local evidence. |
| Six adaptive blocks | Conv1x1 32→64 → GELU → depthwise Conv3x3 64→64 → GELU → Conv1x1 64→32; FiLM 48→64; squeeze gate 32→8→32 sigmoid; learnable residual scale initialized 0.1 | Conditioned residual refinement | Expansion increases channel mixing; depthwise convolution lowers MACs; small initial residual scale keeps early training near identity. |
| Fusion | concatenate 6×32 → Conv1x1 192→32 → GELU → Conv3x3 32→32; add \(f_0\) | Reuses all block depths | Dense history improves feature reuse without a large transformer. |
| Native reconstruction | Conv3x3 32→27 → PixelShuffle(3) | RGB residual at 3H×3W | 27=3×3² channels gives exact single-stage x3 rearrangement without cascaded rounding. |
| Learned observation | Linear 48→49 → softmax, producing a positive normalized 7x7 kernel; unfold/convolution; average-pool3 | \(D_z(\hat y)\) | Positive unit-sum kernel behaves like a plausible blur and avoids unconstrained amplification. |
| Consistency gate | Linear 48→16 → GELU → Linear16→1 → sigmoid, multiplied by 0.25 | \(\alpha\in[0,0.25]\) | Correction bounded hai; LR evidence model output ko dominate nahi kar sakta. |
| Uncertainty head | Conv3x3 32→16 → GELU → Conv3x3 16→1 → sigmoid → bilinear x3 | Spatial risk \(u\in[0,1]\) | One-channel bounded risk map analyst ko questionable areas dikhata hai with small parameter overhead. |

Parameter counts for matched 32-channel variants:

| Variant | DA | Learned observation | Uncertainty | Parameters |
|---|---:|---:|---:|---:|
| Plain | No | No | No | 56,433 |
| Degradation | Yes | No | No | 109,253 |
| Observation | Yes | Yes | No | 112,455 |
| Full | Yes | Yes | Yes | 117,224 |
| Distilled | Yes | Yes | Yes | 117,224 student + frozen 12.2M teacher during training only |

### Forward function

The degradation encoder produces

\[
z=\operatorname{GELU}(W_z\operatorname{GAP}(E_c(x))),\qquad \hat d=\sigma(W_{d2}\operatorname{GELU}(W_{d1}z)).
\]

Each adaptive block follows

\[
h_i=P_{2,i}(\operatorname{GELU}(D_i(\operatorname{GELU}(P_{1,i}(f_{i-1}))))),
\]

\[
\bar h_i=(1+\gamma_i(z))\odot h_i+\beta_i(z),\qquad
f_i=f_{i-1}+s_i\bar h_i\odot\sigma(G_i(\bar h_i)).
\]

Native reconstruction is

\[
\hat y_0=\operatorname{Bicubic}_3(x)+\operatorname{PixelShuffle}_3(C_r(f)).
\]

The learned kernel is positive and normalized:

\[
k_z=\operatorname{softmax}(W_kz),\qquad \sum_{p,q}k_z[p,q]=1,\quad k_z[p,q]\ge 0.
\]

Observation consistency then becomes

\[
\tilde x=D_z(\hat y_0)=\downarrow_3(k_z*\hat y_0),
\]

\[
\alpha(z)=0.25\,\sigma(W_{a2}\operatorname{GELU}(W_{a1}z)),
\]

\[
\hat y=\hat y_0+\alpha(z)\operatorname{Bicubic}_3(x-\tilde x).
\]

Uncertainty is

\[
u=\operatorname{Bilinear}_3\left(\sigma(C_{u2}(\operatorname{GELU}(C_{u1}(f))))\right).
\]

```mermaid
flowchart LR
    X["8-bit LR TIFF tile\nx"] --> N["Normalize to 0..1"]
    N --> ENC["Degradation encoder\nz and d-hat"]
    N --> STEM["3x3 stem"]
    ENC --> BLOCKS["6 FiLM + gated\ndepthwise residual blocks"]
    STEM --> BLOCKS
    BLOCKS --> FUSE["Concatenate + fusion"]
    FUSE --> PS["Conv 32→27 + PixelShuffle x3"]
    N --> BASE["Bicubic x3 base"]
    PS --> Y0["Initial SR y0"]
    BASE --> Y0
    ENC --> K["Softmax 7x7 observation kernel"]
    Y0 --> K
    K --> RED["Re-observed LR"]
    RED --> ERR["x - re-observed LR"]
    ENC --> GATE["Bounded alpha <= 0.25"]
    ERR --> CORR["Upsampled correction"]
    GATE --> CORR
    CORR --> Y["Faithful SR y-hat"]
    FUSE --> UNC["Sigmoid uncertainty map"]
```

Code: [`src/faithsr/model.py`](../../src/faithsr/model.py)

## Activation-function rationale

### GELU

\[
\operatorname{GELU}(a)=a\Phi(a)\approx\frac{a}{2}\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}(a+0.044715a^3)\right]\right).
\]

Why used: GELU hard zero threshold nahi lagata; small negative evidence smoothly retain hoti hai. Transformer teacher bhi GELU family use karta hai, aur compact residual blocks mein smooth gradients stable optimization dete hain. ReLU bhi valid ablation ho sakti hai, lekin current choice ko novelty claim nahi kiya gaya.

### Sigmoid

\[
\sigma(a)=\frac{1}{1+e^{-a}}.
\]

Why used: degradation severities, gates aur uncertainty naturally bounded \([0,1]\) quantities hain. Consistency gate ko further 0.25 se multiply karne ka reason hallucination correction ko conservative rakhna hai. Risk: saturation gradients reduce kar sakti hai; calibration metrics isliye required hain.

### Softmax

\[
\operatorname{softmax}(a_i)=\frac{e^{a_i}}{\sum_j e^{a_j}}.
\]

Why used: learned observation kernel ke weights non-negative aur unit-sum bante hain. Yeh operator ko blur/averaging interpretation deta aur arbitrary signed sharpening kernel se unsupported detail amplification reduce karta hai.

### PixelShuffle

For scale (r=3\), PixelShuffle deterministically rearranges (C r^2\times H\times W\) channels into (C\times rH\times rW\). Is mein learned interpolation nahi hoti; preceding convolution required sub-pixels predict karti hai. It is used because exact x3 output single efficient stage mein milta hai.

References: [GELU](https://arxiv.org/abs/1606.08415), [FiLM](https://arxiv.org/abs/1709.07871), [sub-pixel convolution/PixelShuffle](https://arxiv.org/abs/1609.05158).

## Training objective

The implemented objective is

\[
\mathcal L=\lambda_c\mathcal L_{char}+\lambda_s\mathcal L_{ssim}+\lambda_g\mathcal L_{grad}+\lambda_f\mathcal L_{freq}+\lambda_o\mathcal L_{obs}+\lambda_d\mathcal L_{deg}+\lambda_u\mathcal L_{unc}+\lambda_t\mathcal L_{distill}.
\]

| Term | Mathematical definition | Why it exists |
|---|---|---|
| Charbonnier | \(\mathbb E\sqrt{(\hat y-y)^2+\epsilon^2}\), \(\epsilon=10^{-3}\) | Robust pixel fidelity; large outliers MSE se kam dominate karte hain. |
| SSIM | \(1-\operatorname{SSIM}(\hat y,y)\) using 11x11 local statistics | Local luminance/contrast/structure preserve karta hai. |
| Gradient | \(\|S(\hat y)-S(y)\|_1\), Sobel (S\) | Edges recover karta, but false-edge metric ke sath monitor hota hai. |
| Frequency | \(\||\mathcal F(\hat y)|-|\mathcal F(y)||_1\) | Spatial loss ke sath frequency-energy mismatch reduce karta hai. |
| Observation | smooth-L1 \(D_z(\hat y),x\) | Output ko actual LR measurement se consistent rakhta hai. |
| Degradation | smooth-L1 \((\hat d,d)\) | Encoder ko identifiable synthetic factors learn karwata hai. |
| Uncertainty | smooth-L1 \((u,\operatorname{mean}_c|\operatorname{stopgrad}(\hat y-y)|)\) | Risk map ko observed error magnitude ke sath calibrate karta hai. |
| Distillation | smooth-L1 \((\hat y,t_3(x))\) | Pretrained spatial prior student mein transfer hota, teacher deployment mein required nahi. |

EuroSAT/Sentinel full weights are (1.0,0.1,0.03,0.02,0.05,0.02,0.02,0.0\). Distilled run sets \(\lambda_t=0.05\). These are configuration values, universal constants nahi.

Code: [`src/faithsr/losses.py`](../../src/faithsr/losses.py). SSIM reference: [Wang et al., 2004](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf). Realistic-degradation motivation: [Real-ESRGAN](https://arxiv.org/abs/2107.10833).

## Realistic degradation curriculum and replay manifest

Every synthetic LR observation records blur, Gaussian/Poisson noise severity, compression, aliasing mode, kernel axes/angle, JPEG quality, kernel family, MTF-at-Nyquist target, severity cap, and random seed in `degradations.jsonl`. The training record is append-only by epoch and its SHA-256 is written into `run.json`; this makes an individual LR sample reconstructable rather than merely statistically similar.

For an MTF-inspired Gaussian point-spread function, the implementation uses

\[
\operatorname{MTF}(f)=\exp(-2\pi^2\sigma^2 f^2),\qquad
\sigma=\frac{\sqrt{-2\ln m_N}}{\pi}
\]

at Nyquist frequency \(f=0.5\), where \(m_N\) is the sampled MTF target. An anisotropic axis ratio and angle then represent directional blur. This is a controlled Gaussian MTF approximation, not a claim that one generic kernel reproduces a named satellite sensor.

The epoch severity cap follows

\[
s_e=s_0+(1-s_0)\min\left(1,\frac{e-1}{E_c-1}\right),
\]

where \(s_0\) is the starting severity and \(E_c\) is the configured curriculum duration. Early epochs therefore learn the exact x3 inverse problem under milder corruptions before the full blur/noise/compression/aliasing range is exposed. Validation always uses the frozen full range and deterministic per-image seeds.

```mermaid
flowchart LR
    HR["Held-out/train HR patch"] --> K["Gaussian or MTF-Gaussian PSF"]
    K --> DS["Antialiased exact 3x decimation"]
    DS --> N["Gaussian + conditional Poisson noise"]
    N --> C["JPEG/compression simulation"]
    C --> Q["Clip + uint8 quantization"]
    Q --> LR["Observed LR"]
    K --> M["JSONL parameters + seed"]
    DS --> M
    N --> M
    C --> M
    Q --> M
```

Why added after earlier experiments: EuroSAT v8 and Sentinel v1 preserved only validation degradation details in comparison CSVs. The new manifest closes the training-provenance gap and the curriculum makes the intended easy-to-severe schedule executable. Historical metrics remain under their original generator and are not retroactively relabelled.

## EuroSAT native-3x experiments

### EuroSAT preliminary v8

Why run: native x3 code, deterministic degradation, AMP training, checkpointing aur uncertainty association ko affordable public 8-bit RGB data par end-to-end verify karna tha. UC Merced v0.1 x4 result ko directly reuse nahi kiya ja sakta tha because scale/operator/loss changed.

Connection: v0.1 Full ke fixed bicubic operator ko learned positive 7x7 kernel se replace kiya; degradation target 3 se 4 factors hua; uncertainty head add hui; parameter count 335,541 se 117,224 hua.

Result: 30 epochs, best epoch 25, PSNR 30.6966 dB, SSIM 0.84078, false-edge rate 0.27991, observation L1 0.05491, uncertainty/error correlation 0.28643. Exact run caveats and hashes [`experiment-log.md`](experiment-log.md) mein hain.

### Kaggle engineering-run lineage (v1-v8)

| Kernel version | Experiment/need | Previous run se connection | Outcome and lesson |
|---:|---|---|---|
| 1 | Private Git repository se package clone karna | First remote reproduction attempt | Authentication unavailable; source ko secret-free private Kaggle dataset banana pada. |
| 2 | Source dataset mount use karna | v1 clone failure removed | ZIP/unpacked mount assumption wrong; mount discovery required. |
| 3 | Correct source mount | v2 packaging fix | EuroSAT path hard-coded tha; dataset-root discovery required. |
| 4 | Dynamic data path | v3 path fix | Mounted source ko pip build karna unreliable tha; direct `PYTHONPATH` use hua. |
| 5 | Direct source import | v4 install fix | Tesla P100 (sm60) preinstalled PyTorch build se incompatible; compatible CUDA wheel pin hua. |
| 6 | P100-compatible PyTorch | v5 architecture fix | BCE uncertainty loss AMP-safe nahi tha; smooth-L1 calibration target use hua. |
| 7 | AMP-safe uncertainty | v6 loss fix | Validation batch size 1 ne runtime waste ki; deterministic batched validation add hui. |
| 8 | Batched deterministic validation | v7 efficiency fix | Complete 30-epoch run; checkpoint, split, metrics and logs preserved. |

Why record failed versions: reproducibility sirf successful command nahi hoti. Yeh lineage batati hai ke final kernel ke environment pins aur mount-discovery code arbitrary nahi, observed failures ke direct fixes hain.

### Matched interpolation evaluation v3

Why run: standalone model number se superiority prove nahi hoti. Same 6,000 deterministic samples/degradations par nearest, bilinear, bicubic aur saved FaithSR checkpoint compare kiye.

Connection: v8 checkpoint freeze karne ke baad separate kernel mein evaluation hui, isliye test-time checkpoint selection nahi hua.

| Method | PSNR | SSIM | False-edge rate | SAM (rad) | ERGAS |
|---|---:|---:|---:|---:|---:|
| Nearest | 21.3032 | 0.51278 | 0.68459 | 0.14767 | 9.08825 |
| Bilinear | 23.7156 | 0.58633 | 0.68736 | 0.10980 | 6.86787 |
| Bicubic | 22.7390 | 0.55729 | 0.69036 | 0.12514 | 7.70141 |
| FaithSR-3X | **30.6825** | **0.84060** | **0.28037** | **0.03489** | **3.18576** |

Interpretation: same synthetic generator par FaithSR interpolation se strong better hai. Yeh real unknown sensor degradation, pretrained baseline, ya SUPARCO result nahi.

Artifacts: [`artifacts/results/faithsr3x/eurosat_matched_v3/`](../../artifacts/results/faithsr3x/eurosat_matched_v3/)

## Sentinel-2 Delhi geographic ablation experiment

Status: four matched custom variants completed in Kaggle kernel v1. Distillation attempts v1/v2 failed for documented configuration/autograd reasons; corrected v3 completed with a compatible pinned environment. Historical failures and final artifacts are both retained.

Dataset role: CC0 Sentinel-2 RGB display patches, 128x128 at reported 10 m/pixel, coordinate-named over Delhi NCR. It is more geographically explicit than EuroSAT but does not contain native L2A DN/quantification metadata, so it is not a replacement for the planned reflectance-controlled L2A experiment.

Split function: for coordinate ((\varphi,\lambda)\), group

\[
g(\varphi,\lambda)=\left(\left\lfloor\frac{\varphi}{0.1}\right\rfloor,\left\lfloor\frac{\lambda}{0.1}\right\rfloor\right).
\]

Entire 0.1-degree cells are assigned to train or validation. Adjacent patches from one cell cannot leak across the split.

### Matched ablation questions

| Variant | Change from previous | Why it is necessary | Expected evidence |
|---|---|---|---|
| Plain | No encoder, FiLM, observation, uncertainty | Minimum custom native-x3 control. | Backbone capacity baseline. |
| Degradation | Add encoder + FiLM | Isolate whether degradation conditioning itself helps. | Plain→DA paired gain/loss. |
| Observation | Add learned re-degradation and bounded correction | Test the mechanism motivated by v0.1 AOC failure. | DA→Observation fidelity/false-edge change. |
| Full | Add uncertainty head/loss | Test whether risk output can be added without fidelity regression. | Observation→Full plus correlation/calibration. |
| Distilled | Add frozen pinned Swin2SR target, same deployed student | Test whether pretrained knowledge personalizes the custom model rather than replacing it. | Full→Distilled fidelity trade-off. |
| Frozen Swin2SR | 12.2M pretrained x4 then explicit x3 resize | Strong external reference. | Proposed student versus pretrained model. |

```mermaid
flowchart TD
    DATA["Sentinel-2 Delhi RGB\ncoordinate tiles"] --> GRID["0.1-degree geographic grouping"]
    GRID --> TRAIN["Frozen train cells"]
    GRID --> VAL["Frozen validation cells"]
    TRAIN --> P["Plain"]
    TRAIN --> D["+ Degradation"]
    TRAIN --> O["+ Observation"]
    TRAIN --> F["+ Uncertainty = Full"]
    TRAIN --> T["+ Swin2SR distillation"]
    VAL --> CMP["Matched metrics"]
    P --> CMP
    D --> CMP
    O --> CMP
    F --> CMP
    T --> CMP
    VAL --> SW["Frozen Swin2SR baseline"]
    SW --> CMP
```

Kernel: `muhammadhassan2/faithsr-3x-sentinel-delhi-ablation`. Config: [`configs/sentinel_delhi_kaggle.yaml`](../../configs/sentinel_delhi_kaggle.yaml).

## Native Sentinel-2 L2A radiometric experiment

Why run: Delhi tiles are useful geographic RGB evidence but omit native L2A DN, scale/offset and acquisition provenance. This experiment closes that gap while still targeting the supervisor-confirmed 8-bit TIFF interface.

For STAC-provided digital number (d_c), scale (s_c), offset (o_c), and frozen display limits (a=0,b=0.3), each B04/B03/B02 channel is mapped by

\[
r_c=s_c d_c+o_c,\qquad
u_c=\operatorname{round}\left(255\,\operatorname{clip}\left(\frac{r_c-a}{b-a},0,1\right)\right).
\]

This mapping is a documented derived 8-bit product, not a claim that native Sentinel data are intrinsically 8-bit. Source item ID, canonical STAC item hash, band URLs, acquisition time, cloud cover, CRS, transform, source/window dimensions, scale, offset and clipping limits are preserved in JSON and TIFF tags. Dataset masks exclude nodata. The L2A SCL classes 2, 4, 5, 6 and 7 define mechanically selected usable overview windows; cloud/cloud-shadow/snow/nodata classes are excluded. An eligible overview block requires at least 95% accepted SCL pixels and 55% pixels above 0.01 reflectance, then maximum red-channel P90-P10 range wins. Every final uint8 tile also requires at least 75% nonzero pixels and mean channel standard deviation at least 2.0. These thresholds use input statistics only and are frozen before training.

```mermaid
flowchart LR
    STAC["Element84 Sentinel-2 L2A STAC"] --> META["Item ID, date, CRS, scale/offset"]
    STAC --> RGB["10 m B04/B03/B02 COG windows"]
    STAC --> SCL["Scene Classification Layer"]
    SCL --> PICK["Mechanically select valid window"]
    RGB --> REF["DN to surface reflectance"]
    META --> REF
    REF --> U8["Frozen [0,0.3] clip to uint8"]
    PICK --> U8
    U8 --> TILE["192x192 georeferenced TIFF tiles"]
    TILE --> SCENE["Whole-scene train/validation split"]
    SCENE --> EXP["Plain / Uncertainty-only / Full"]
    EXP --> EVID["Metrics + saved held-out preview"]
```

The native comparison isolates uncertainty explicitly:

| Variant | Active paths | Mathematical change | Why needed |
|---|---|---|---|
| Plain | GELU residual backbone + PixelShuffle(3) | \(\hat y=P_3(C(h_L))+B_3(x)\) | Minimum custom exact-3x control. |
| Uncertainty-only | Plain + sigmoid risk head | \(u=\sigma(C_u(h_L))\) and heteroscedastic calibration term | Measures risk-head cost/benefit without DA/AOC confounding. |
| Full | Encoder/FiLM + learned observation correction + uncertainty | Full forward and loss defined above | Tests the complete personalized hypothesis on native radiometric provenance. |

All three use the same six-scene candidate design, accepted v6 manifest, seed 0, 60 epochs, MTF-aware mixed degradation curriculum, scene-parent split, best-validation-PSNR selection, and per-image interpolation comparison. The reported tile count comes only from the accepted manifest after content filtering. Each best-checkpoint preview uses the held-out sample at median HR spatial standard deviation, selected before training without model scores. GELU remains the backbone activation for smooth gating; sigmoid is used only where a bounded 0-to-1 uncertainty/risk field is required; PixelShuffle(3) makes the output scale exactly 3 rather than resizing an x4 pretrained output.

### Native L2A v6 accepted mixed/negative result

The final content screen retained 86 tiles from six scenes. Whole-scene separation produced 39 training and 47 held-out validation images in three scene clusters. v4 was rejected because comparison used legacy degradation defaults; v5 was rejected because comparison still used seed 42. v6 records seed 0, crop 192, MTF probability 0.5 and mixture probability 0.75, and its non-LPIPS bicubic metrics exactly match the independent pretrained experiment. Per-image means are:

| Method | Params | Best epoch | PSNR | SSIM | False-edge | SAM (rad) | ERGAS | LPIPS | UQ corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0 | — | **23.41633** | 0.59057 | **0.13229** | **0.40417** | **28.04499** | 0.62166 | — |
| Plain | 56,433 | 58 | 23.13413 | 0.59140 | 0.13601 | 0.41702 | 29.12315 | 0.53151 | — |
| Uncertainty-only | 61,202 | 56 | 23.13498 | **0.59275** | 0.13560 | 0.41663 | 29.08632 | **0.53138** | -0.25121 |
| Full | 117,224 | 60 | 23.05308 | 0.58783 | 0.13795 | 0.42096 | 29.37433 | 0.54940 | -0.26663 |

Scene-cluster intervals prevent unequal scene sizes from silently dominating inference. Plain-minus-bicubic PSNR is -0.26044 dB (95% CI -0.33753 to -0.18336) and LPIPS is -0.09189 (-0.10172 to -0.08206). Full-minus-bicubic PSNR is -0.33530 dB (-0.43653 to -0.23407), false-edge is +0.00496 (+0.00019 to +0.00972), and LPIPS is -0.07370 (-0.08085 to -0.06655). Thus the compact network has a perceptual trade-off but does not beat bicubic fidelity on this stricter native corpus; the complete personalization paths hurt more than they help. Negative uncertainty-error correlation rejects the current calibrated-risk interpretation. The tiny 39-tile training side and native radiometric/domain shift are plausible constraints, not excuses or proven causes; held-out scenes will not be used for post-hoc tuning.

Artifacts: [`sentinel_l2a_native_v6`](../../artifacts/training_runs/faithsr3x/sentinel_l2a_native_v6/). The automatic previews directly print bicubic and model PSNR, and `artifact-provenance.json` freezes hashes for key evidence.

### Fully fine-tuned pretrained control on native L2A

This control answers whether personalization of a pretrained model works when every weight, rather than only a wrapper or custom student, adapts to the native training split. For pinned Swin2SR x4 function (S_{\theta}), the exact comparison output is

\[
\hat y=\mathcal R_{3H,3W}(S_{\theta}(x)),\qquad
\theta^*=\arg\min_{\theta}\sum_i\operatorname{SmoothL1}(\hat y_i,y_i).
\]

All 12,239,283 parameters are trainable for three AdamW epochs at learning rate (10^{-5}). Shifted-window attention captures nonlocal structure; LayerNorm stabilizes transformer blocks; GELU supplies smooth MLP gating; residual connections preserve low-frequency evidence; the inherited x4 PixelShuffle head reconstructs first, and a separately declared antialiased bicubic operator produces exactly x3. This last resize is necessary because the immutable public checkpoint is x4 and prevents a false native-x3 claim.

```mermaid
flowchart LR
    LR["Native-protocol LR"] --> EMB["Convolutional embedding"]
    EMB --> SW["Residual shifted-window SwinV2 groups\nLayerNorm + attention + GELU MLP"]
    SW --> PX["Inherited PixelShuffle x4"]
    PX --> R3["Declared antialiased resize to x3"]
    R3 --> L["Smooth-L1 to held-out-protocol HR"]
```

Frozen Swin2SR reaches 21.92152 dB PSNR, 0.54311 SSIM and 0.47065 LPIPS. Full fine-tuning reaches 23.77746 dB, 0.61740 and 0.55062 versus bicubic 23.41633 dB, 0.59057 and 0.62166. Fine-tuned scene-cluster deltas are +0.33646 dB PSNR (95% CI +0.04417 to +0.62876), +0.02883 SSIM, -0.00821 false-edge, -0.02050 rad SAM and -0.07080 LPIPS. This is strong evidence that domain/task personalization matters. It is not evidence that a 12.24M x4-plus-resize model is the project's compact native-x3 contribution; it is roughly 104.4x larger than Full FaithSR. These are held-out validation metrics because the same split selected epoch 3, not an untouched final test.

Artifacts and complete generated model card: [`swin2sr_full_finetune_native_l2a_v3`](../../artifacts/training_runs/faithsr3x/swin2sr_full_finetune_native_l2a_v3/). References: Conde et al., [Swin2SR](https://arxiv.org/abs/2209.11345), and the [immutable checkpoint configuration](https://huggingface.co/caidas/swin2SR-classical-sr-x4-64/blob/c69ef3e2d2ac5777ff4a9f2f5afcd86d20604b7e/config.json).

### Native-x4 diagnostic and architecture decision

The corrected minimum-3x requirement permits the direct native mapping

\[
\hat y_4=S_{\theta}(x),
\]

without the matched-x3 resize used above. The accepted v2 run keeps every pretrained layer trainable and directly supervises the x4 PixelShuffle output with Smooth-L1. Its shallow convolution, shifted-window SwinV2 attention groups, LayerNorm, GELU MLPs, residual convolutions and reconstruction head are inherited unchanged so that personalization remains attributable to domain fine-tuning rather than an undocumented architecture mutation.

```mermaid
flowchart LR
    LR["Sentinel LR"] --> SW["Personalized Swin2SR x4<br/>12.24M parameters"]
    SW --> HR["Native x4 SR"]
    HR --> FID["Fidelity stress grid"]
    HR --> LAT["512-to-2048 latency"]
    FID --> DEC["Compression weakness measured"]
    LAT --> DEC2["One-second gate failed"]
    DEC --> STUD["Compression-aware lightweight x4 student"]
    DEC2 --> STUD
```

Across 329 paired stress cases, fine-tuning improves PSNR by +0.50377 dB (95% scene-cluster CI +0.40189 to +0.60565), SSIM by +0.04228, false-edge rate by -0.01460, and LPIPS by -0.08691 versus bicubic. Compression-heavy PSNR is only +0.0193 dB with a confidence interval crossing zero. FP32 model-only P95 is 3.419 seconds for 512x512 input on Tesla P100, so the teacher is too slow for the deployment target.

The next model is selected from these failures. A compact native-x4 student will receive full-schedule output/feature supervision from the accepted personalized teacher, while an estimated degradation vector and bounded observation correction address compression robustness. This is not a repeat of the neutral earlier x3 distillation: teacher and target are now native x4, the teacher is Sentinel-personalized rather than frozen generic, supervision spans the training schedule, and compression cases are explicitly stratified. LoRA is insufficient for the latency goal because it does not remove the teacher backbone at inference. Mamba is deferred because the diagnostic did not establish long-range structure as the dominant failure.

Detailed protocol, per-profile table, latency, visuals and decision: [`swin2sr-native-x4-diagnostic-results.md`](swin2sr-native-x4-diagnostic-results.md). Accepted artifacts: [`swin2sr_native_x4_diagnostic_v2`](../../artifacts/training_runs/faithsr3x/swin2sr_native_x4_diagnostic_v2/).

### Sentinel v1 completed results

The frozen split contains 7,488 training images in 58 coordinate-grid groups and 1,728 validation images in 14 disjoint groups; group overlap is zero. All variants use seed 42, 30 epochs, identical synthetic degradation/evaluation, and best-validation-PSNR checkpoint selection.

| Variant | Best epoch | Params | PSNR | SSIM | False-edge | SAM (rad) | ERGAS | UQ corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain | 25 | 56,433 | 24.1724 | 0.65765 | 0.13127 | 0.07608 | 8.37479 | — |
| + Degradation | 22 | 109,253 | **24.1976** | 0.65711 | **0.12943** | **0.07491** | **8.34914** | — |
| + Observation | 28 | 112,455 | 24.1897 | **0.65796** | 0.13164 | 0.07513 | 8.35818 | — |
| + Uncertainty (Full) | 26 | 117,224 | 24.1911 | 0.65753 | 0.13142 | 0.07520 | 8.35577 | 0.19524 |
| Bicubic | — | 0 | 22.3652 | 0.56177 | 0.16018 | 0.16274 | 10.49418 | — |

The table uses authoritative per-image means from exact-evaluation v2. Earlier v1 batch-level PSNR was lower because PSNR is nonlinear and had first been computed after pooling each batch; that aggregate is retained only as historical evidence.

Paired cluster-robust results sharpen the interpretation. DA minus Plain PSNR is +0.02594 dB (95% CI +0.02112 to +0.03076) and false-edge change is -0.00192 (-0.00214 to -0.00169): small but consistent gains. Observation minus DA trades +0.00078 SSIM for -0.00826 dB PSNR and +0.00227 false-edge; thus the current learned observation mechanism is not supported as an across-metric improvement. Full minus Observation adds +0.00159 dB PSNR and -0.00023 false-edge but -0.00041 SSIM. Full still beats bicubic by a geographic-cluster mean +1.84378 dB (95% CI +1.69044 to +1.99712), reduces false-edge by 0.02908, and its uncertainty has positive per-image error association (0.19524). Unqualified claims that every component improves every metric are rejected.

Each completed run has its own generated `model-card.md`, exact YAML, metrics, split, comparison and checkpoint under [`artifacts/training_runs/faithsr3x/sentinel_delhi_ablation_v1_partial/`](../../artifacts/training_runs/faithsr3x/sentinel_delhi_ablation_v1_partial/). Per-image tables and paired component intervals are under [`artifacts/results/faithsr3x/sentinel_exact_v2/`](../../artifacts/results/faithsr3x/sentinel_exact_v2/).

### Sentinel v1 failure and corrected personalization design

The run reached the `distilled` command only after the four rows above were complete. Importing Swin2SR then failed with `operator torchvision::nms does not exist`: pinned Torch 2.5.1 was paired with Kaggle's preinstalled TorchVision 0.25.0, which expected Torch 2.10.0. The follow-up pins the compatible TorchVision 0.20.1 build and records this environment decision in code and provenance.

The corrected distilled model keeps the same 117,224-parameter deployed architecture. A frozen, revision-pinned Swin2SR produces targets in micro-batches of two for the first three epochs:

\[
\mathcal L_e=\mathcal L_{FaithSR}+\mathbf 1[e\le3]\,0.05\,\operatorname{smoothL1}(\hat y,t_3(x)).
\]

For epochs 4–30, only data-driven FaithSR losses remain. Why: the short teacher phase transfers a pretrained image prior, while the longer teacher-free phase personalizes the student to the Sentinel degradation/data distribution and prevents the x4 natural-image teacher from dominating. Teacher micro-batching changes memory scheduling only, not the mathematical target. Kernel: `muhammadhassan2/faithsr-3x-sentinel-distillation`.

## Downstream WorldCover land-cover probe

Why run: PSNR aur edge metrics analyst utility ka complete substitute nahi. A separate 53,335-parameter classifier was trained only on original HR RGB tiles, frozen, and then applied unchanged to HR, bicubic-x3 and FaithSR-x3 versions of the untouched main validation groups.

For each coordinate-named RGB tile, the target is the majority non-zero ESA WorldCover class in the aligned 126x126 raster window centred at that latitude/longitude. The original 58 training groups were split into 49 classifier-fit and 9 calibration groups; the 14 main validation groups remained final test. Two test samples whose rare class was absent from fit were excluded and reported, leaving 1,726.

The probe computes

\[
h_i=\operatorname{GELU}(C^{(i)}_{3\times3,s=2}(h_{i-1})),\quad i=1,2,3,
\]

with channels 3→24→48→96, then (v_c=\operatorname{mean}_{p,q}h_3[c,p,q]), logits (z=Wv+b), and cross-entropy. GELU supplies smooth feature transformations; global average pooling avoids a coordinate-specific dense head. Exact equations, rationale and Mermaid diagram are in the [auxiliary model card](../../artifacts/results/faithsr3x/downstream_landcover_v2/model-card.md).

| Input to frozen probe | Accuracy | Macro-F1 |
|---|---:|---:|
| Original HR reference | 0.65469 | 0.23651 |
| Bicubic x3 | 0.65238 | 0.23543 |
| FaithSR-3X | **0.66512** | **0.24084** |

FaithSR minus bicubic paired accuracy gain is +0.01540 across geographic clusters (95% CI +0.00658 to +0.02422). The HR reference itself is not significantly different from bicubic (+0.00347, CI -0.00470 to +0.01163), while FaithSR slightly exceeds both for this frozen probe.

Limitation: calibration macro-F1 is only 0.1632 and final HR macro-F1 0.2365 because labels are imbalanced patch-majority categories and the probe is deliberately small. Therefore this is evidence of modest semantic retention/improvement under one downstream protocol, not a claim of high-quality land-cover classification or newly sensed 3.33 m labels. Artifacts: [`downstream_landcover_v2`](../../artifacts/results/faithsr3x/downstream_landcover_v2/).

## GeoTIFF inference and uncertainty products

Large raster processing uses overlapping tiles, Hann blending and disk-backed sum/weight rasters:

\[
\hat y(p)=\frac{\sum_i w_i(p)\hat y_i(p)}{\max(\sum_iw_i(p),\epsilon)}.
\]

Only one model tile and one final output block are held in memory. CRS/bounds remain fixed; width/height become 3x and affine pixel spacing becomes one third. Output is clipped, rounded and written as uint8. Uncertainty is a separate float32 GeoTIFF on the identical grid; it is not mixed into the scientific image bands.

```mermaid
flowchart LR
    TIFF["8-bit TIFF/GeoTIFF"] --> META["Validate + capture metadata"]
    TIFF --> WIN["Overlapping raster windows"]
    WIN --> GPU["FaithSR tile inference"]
    GPU --> DS["Disk-backed weighted sums"]
    DS --> OUT["Windowed uint8 finalization"]
    META --> OUT
    GPU --> RISK["Disk-backed uncertainty sums"]
    RISK --> UTIFF["Float32 uncertainty GeoTIFF"]
    OUT --> AUDIT["Hashes + timings + metadata audit JSON"]
```

Why Hann blending: overlapping predictions ke borders par convolution/padding artifacts zyada hote hain. Smooth weights centre evidence ko prefer karte aur hard tile seams avoid karte hain. Disk accumulator ka reason representative large rasters par full 3x tensor RAM/GPU allocation remove karna hai.

### Deployment benchmark v5 result

The first complete 512x512 TIFF benchmark measured 52.96 ms model-only P95 but 1266.34 ms full-pipeline P95 on a Tesla P100. Mean finalization/write time was 1077.23 ms, compared with 78.88 ms inference and 82.53 ms disk accumulation. Therefore the complete Deflate pipeline failed the provisional one-second threshold even though the neural model passed comfortably.

Connection and need: model-only v3 timing could not answer whether the analyst actually receives a TIFF within one second. v5 added input read, disk-backed blending, uint8 conversion, lossless compression, uncertainty GeoTIFF, masks, metadata and audit hashes. The next matched experiment changes only the lossless compression strategy (input/Deflate versus LZW versus none) to isolate I/O cost; pixel/model behavior remains fixed.

Code: [`src/faithsr/inference.py`](../../src/faithsr/inference.py), [`src/faithsr/geotiff.py`](../../src/faithsr/geotiff.py).

### Deployment compression ablation v6

Why run: v5 isolated finalization/write as the bottleneck. Version 6 changes only the lossless output compression while holding checkpoint, 512x512 fixture, tiling and P100 hardware fixed.

| Compression | Mean end-to-end | Median | P95 | Mean finalize/write | Result vs 1 s |
|---|---:|---:|---:|---:|---|
| Input/Deflate | 590.20 ms | 513.89 ms | 743.43 ms | 431.33 ms | Pass |
| LZW | 626.85 ms | 598.31 ms | 697.34 ms | 467.92 ms | Pass |
| None | **381.45 ms** | **343.21 ms** | **460.42 ms** | **222.11 ms** | Pass |

Interpretation: all three reruns passed one second, while uncompressed output was fastest. The absolute Deflate improvement from v5 (1266.34 to 743.43 ms P95) despite identical logical settings shows three-run I/O timing variability; therefore compression ordering is useful engineering evidence but not a universal latency guarantee. Default lossless compression remains safer for storage, while `none` is an explicit speed/storage option. Immutable artifacts: [`deployment_benchmark_v6`](../../artifacts/results/faithsr3x/deployment_benchmark_v6/).

### Verified ONNX export

Why run: an exported graph that cannot execute at non-example shapes is not a deployment artifact. The frozen EuroSAT v8 model was exported at ONNX opset 18 with batch one and dynamic spatial axes, structurally checked, and compared with PyTorch using ONNX Runtime at 64x64 and 67x69 inputs. Both outputs were exact 3x; maximum absolute floating-point difference was (1.04\times10^{-5}). The SHA-256-recorded artifact and verification sidecar are in [`artifacts/exports/`](../../artifacts/exports/). FP16 optimization remains a separate matched experiment because format conversion alone does not establish speed or fidelity.

### FP32 versus FP16 deployment precision

Why run: ONNX verification established graph portability, but it did not show whether reduced precision is faster or whether 8-bit deliverables remain stable. This matched P100 experiment held the EuroSAT v8 checkpoint, 512x512 input, tiling and uncompressed output fixed; only CUDA inference precision changed.

| Precision | Model mean | Model P95 | Peak VRAM | TIFF mean | TIFF P95 |
|---|---:|---:|---:|---:|---:|
| FP32 | 52.80 ms | 53.03 ms | 3,200,190,976 B | 357.73 ms | 497.86 ms |
| FP16 autocast | **51.91 ms** | **52.14 ms** | **3,067,272,704 B** | **348.68 ms** | **450.99 ms** |

FP16 reduced model P95 by 1.67% and end-to-end P95 by 9.42%. After conversion to the required uint8 GeoTIFF, the maximum difference was one digital number (DN), mean absolute difference 0.0679 DN, 6.79% of channel pixels changed by one DN, and metadata was identical. Therefore FP16 is an optional CUDA deployment mode, not a claim of mathematically identical floating-point output. INT8/TensorRT remains hardware-specific future work because an uncalibrated quantizer could damage the project’s fidelity objective. Immutable results and hashes: [`precision_benchmark_v1`](../../artifacts/results/faithsr3x/precision_benchmark_v1/).

### Swin2SR teacher distillation personalization

Why run: the project must demonstrate a defensible use of pretrained knowledge rather than relabel a downloaded network. A frozen published Swin2SR classical-SR x4 teacher generated targets during epochs 1–3; the trainable 117,224-parameter FaithSR x3 student then continued through epoch 30 under the same degradation, observation-consistency, uncertainty, Charbonnier, SSIM, gradient and frequency objectives as the full Sentinel model.

The teacher mapping is

\[
T_3(x)=\operatorname{Resize}_{3/4}(T_{\text{Swin2SR-x4}}(x)),
\qquad
\mathcal L_{KD}=\lVert \hat y-T_3(x)\rVert_1,
\]

and epochs 1–3 optimize \(\mathcal L_{full}+0.05\mathcal L_{KD}\). Resizing is explicit because this experiment compares an intrinsically x4 public teacher against an x3 student under the same x3 target protocol. It is not imposed by the corrected SUPARCO requirement, which accepts any scale of at least 3x. GELU is used inside both the compact student residual blocks and the teacher’s MLPs because its smooth gating supports gradient flow; the student ends with a 27-channel convolution and PixelShuffle(3), which deterministically rearranges channels into exactly 3x spatial dimensions.

```mermaid
flowchart LR
    LR["Synthetic LR Sentinel patch"] --> S["Trainable FaithSR x3 student"]
    LR --> T["Frozen pretrained Swin2SR x4"]
    T --> R["Resize x4 output to exact x3"]
    S --> KD["L1 distillation loss, epochs 1-3"]
    R --> KD
    S --> DATA["Data, observation and uncertainty losses"]
    KD --> OPT["Update student only"]
    DATA --> OPT
```

On 1,728 images from 14 untouched geographic groups, the distilled student achieved 24.19109 dB PSNR, 0.65754 SSIM, 0.13142 false-edge rate, 0.07522 rad SAM and 8.35574 ERGAS. Its PSNR versus bicubic improved by 1.84381 dB (cluster-robust 95% CI 1.69050–1.99713). However, it was effectively tied with the same architecture trained without distillation (24.19106 dB; difference about +0.00004 dB), so this run proves the personalization mechanism and reproducibility, not a meaningful teacher-derived accuracy gain.

The unadapted x4 teacher resized to x3 performed poorly on this matched Sentinel degradation (19.67141 dB PSNR, 0.45800 SSIM, 0.16445 false-edge rate), below bicubic. This is direct evidence that a generic pretrained model cannot be presented as the final solution: scale mismatch and domain shift matter, while the personalized compact x3 student is the deployable artifact. Artifacts: [`sentinel_distillation_v3`](../../artifacts/training_runs/faithsr3x/sentinel_distillation_v3/).

## Reference list

1. Conde et al., “Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration,” 2022: https://arxiv.org/abs/2209.11345
2. Shi et al., “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network,” 2016: https://arxiv.org/abs/1609.05158
3. Hendrycks and Gimpel, “Gaussian Error Linear Units (GELUs),” 2016: https://arxiv.org/abs/1606.08415
4. Perez et al., “FiLM: Visual Reasoning with a General Conditioning Layer,” 2017: https://arxiv.org/abs/1709.07871
5. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” 2004: https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
6. Wang et al., “Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,” 2021: https://arxiv.org/abs/2107.10833
7. Ngambenjavichaikul et al., “Development of a Super-Resolution Satellite Image Processing System on an Edge AI Platform,” ISMAC 2025. Local supplied text: [`docs/references/edge-ai-sr-platform-paper.md`](../references/edge-ai-sr-platform-paper.md).
8. Delhi Airshed Sentinel-2 RGB and Landcover dataset (CC0): https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed
9. Element84 Earth Search Sentinel-2 L2A STAC collection: https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a
