# Downstream Land-cover Probe

## Why reconstruction metrics were not enough

PSNR and SSIM do not directly answer whether enhanced imagery preserves information useful to an analyst. This experiment trained a small classifier only on original HR tiles, froze it, then evaluated HR, bicubic and FaithSR images on untouched geographic groups.

## Probe function and architecture

Input is **B RGB images**. Three 3×3 stride-2 convolutions change channels 3→24→48→96. GELU follows each convolution. Global average pooling converts each final feature map to one value, then a linear layer produces land-cover logits. Cross-entropy trains class prediction.

$$
\mathbf{h}_3=\operatorname{GELU}\!\left(C^{96}_{3\times3,s=2}\!\left(
\operatorname{GELU}\!\left(C^{48}_{3\times3,s=2}\!\left(
\operatorname{GELU}(C^{24}_{3\times3,s=2}(\mathbf{x}))\right)\right)\right)\right),
$$

$$
\boldsymbol{\ell}=W\,\operatorname{GAP}(\mathbf{h}_3)+\mathbf{b},
\qquad
\mathcal{L}_{\mathrm{cls}}=-\sum_k y_k\log\operatorname{Softmax}(\boldsymbol{\ell})_k.
$$

$\boldsymbol{\ell}$ contains land-cover logits, GAP is global average pooling, and $y_k$ is the reference class indicator. Code: the frozen probe implementation in `src/faithsr/downstream.py`.

Why: stride-2 convolutions cheaply summarize spatial evidence; GELU supplies smooth non-linearity; global average pooling avoids a coordinate-specific dense image head. The probe has 53,335 parameters and is independent of SR training.

## Result

| Frozen-probe input | Accuracy | Macro-F1 |
|---|---:|---:|
| Original HR | 0.65469 | 0.23651 |
| Bicubic x3 | 0.65238 | 0.23543 |
| FaithSR x3 | **0.66512** | **0.24084** |

FaithSR minus bicubic paired accuracy was +0.01540 with 95% geographic-cluster interval +0.00658 to +0.02422.

## Limitation and decision

The classifier is weak and labels are imbalanced majority WorldCover classes. The result supports modest semantic retention under one probe; it does not prove new subpixel land-cover truth or state-of-the-art classification.

Artifacts and generated model card: [`downstream_landcover_v2`](../../../../artifacts/results/faithsr3x/downstream_landcover_v2/).

References: ESA WorldCover; Hendrycks and Gimpel, *GELU* (2016).
