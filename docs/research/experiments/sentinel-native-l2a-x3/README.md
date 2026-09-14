# Native Sentinel-2 L2A x3 Validation

## Research question

Do the custom components that looked useful on display-scaled Delhi imagery remain reliable on native Sentinel-2 Level-2A reflectance with complete-scene separation?

## Dataset protocol

Six Sentinel-2 L2A scenes were obtained through the Element84 Earth Search STAC API: two each around Karachi, Lahore and Islamabad. B04, B03 and B02 10 m reflectance bands formed RGB, while the SCL layer screened invalid content. The final corpus contained 86 content-screened tiles: 39 train and 47 held-out tiles from three disjoint parent scenes. Seed was 0.

STAC scale and offset were applied first; reflectance was then clipped to 0–0.3 and mapped to uint8 for the declared product. This is a display-ready RGB research product, not the untouched 13-band sensor cube.

## Function and architecture

All variants map **B RGB tiles × H×W** to **B RGB tiles × 3H×3W**. Plain uses convolution, six GELU residual blocks and PixelShuffle(3). Uncertainty-only adds a non-negative risk head. Full adds degradation FiLM, observation correction and uncertainty.

$$
\mathbf{x}\in[0,1]^{B\times3\times H\times W},
\qquad
\hat{\mathbf{y}}=F_{\theta,3}(\mathbf{x})
\in[0,1]^{B\times3\times3H\times3W}.
$$

The native reconstruction before optional correction is

$$
\hat{\mathbf{y}}_0=\operatorname{PixelShuffle}_3(C_R(\mathbf{f}_6)),
$$

where $\mathbf{f}_6$ is the output of six residual blocks and $C_R$ produces $3\times3^2=27$ channels. Full additionally evaluates

$$
\mathcal{L}_{\mathrm{obs}}=\operatorname{SmoothL1}(D_{\mathbf{z}}(\hat{\mathbf{y}}),\mathbf{x}),
$$

where $D_{\mathbf{z}}$ is the learned degradation-conditioned observation operator. Code: `FaithSR3X.forward`, `LearnedObservationOperator.forward`, and `FaithSRObjective.forward`.

The observation term asks: “if the output is degraded back to the LR grid, does it reproduce the actual input?” This discourages unsupported detail. It is a regularizer, not an inference-time pre-degradation step.

## Version audit

- v1: incompatible CUDA architecture; no metric accepted.
- v2: preview CPU/GPU mismatch; no completed metric accepted.
- v3: dark/non-informative windows caused degenerate scores; rejected.
- v4: content gates fixed windows, but evaluator used legacy degradation defaults; rejected.
- v5: evaluator seed mismatch; rejected.
- v6: exact protocol replay and independent bicubic agreement; accepted.

## Accepted v6 result

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 23.41633 | 0.59057 | 0.62166 |
| Plain | 23.13413 | 0.59140 | **0.53151** |
| Uncertainty-only | 23.13498 | 0.59275 | **0.53138** |
| Full | 23.05308 | 0.58783 | 0.54940 |

Plain trails bicubic PSNR by 0.26044 dB but improves LPIPS by 0.09189. Full trails PSNR by 0.33530 dB and slightly worsens false-edge rate. Uncertainty/error correlations were negative, so the risk output is not a calibrated confidence probability.

## Decision and connection

This mixed/negative experiment prevented promotion of the custom x3 model on native data. It motivated a full fine-tune of a strong pretrained model on exactly the same split to test whether the limiting factor was domain adaptation rather than the dataset alone.

Artifacts: [`sentinel_l2a_native_v6`](../../../../artifacts/training_runs/faithsr3x/sentinel_l2a_native_v6/).

References: Element84 Earth Search Sentinel-2 L2A; ESA Sentinel-2 MSI; Real-ESRGAN degradation synthesis (Wang et al., 2021).
