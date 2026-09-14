# Sentinel x3 Teacher Distillation

## Why this experiment was needed

The Delhi ablation did not complete its teacher variant. This follow-up tested whether a public frozen Swin2SR prior gives the compact x3 student a measurable improvement, while keeping the deployed network custom and lightweight.

## Teacher and student function

The teacher accepts an LR RGB tile and produces x4. For matched x3 evaluation only, its output is antialiased-resized to x3:

$$
\mathbf{t}_3=R_{4\rightarrow3}\!\left(T_{\mathrm{Swin2SR},4}(\mathbf{x})\right),
$$

The student directly produces x3:

$$
\hat{\mathbf{y}}_3=S_{\theta,3}(\mathbf{x}),
$$

During epochs 1–3:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{data}}+0.05\,\operatorname{SmoothL1}(\hat{\mathbf{y}}_3,\mathbf{t}_3).
$$

$T$ is the frozen Swin2SR x4 teacher, $R_{4\rightarrow3}$ is declared antialiased resizing, and $S_{\theta,3}$ is the trainable native-x3 student. Code: `Swin2SRTeacher.forward`, `FaithSR3X.forward`, and `FaithSRObjective.forward`.

During epochs 4–30, the teacher term is removed. In plain language, the model briefly learns the pretrained prior and then spends most training adapting to Sentinel data. The teacher is never needed in the deployed app.

Swin2SR uses shifted-window attention, LayerNorm, GELU MLPs and residual paths. The student uses 3×3 convolution, GELU residual blocks and PixelShuffle(3). Smooth-L1 was chosen because it is less sensitive to large outliers than squared error.

## Result

The distilled student reached 24.19109 dB PSNR and 0.65754 SSIM; the matched non-distilled Full student was 24.19106 dB. The difference, about +0.00004 dB, is not material. Generic frozen Swin2SR resized x4→x3 was below bicubic, demonstrating domain/scale mismatch.

## Failures and decision

Version 1 had an invalid Kaggle kernel-source dependency. Version 2 allowed teacher autograd and failed. Version 3 froze teacher gradients and completed. The experiment proves a reproducible personalization mechanism, but rejects a claim that this particular short distillation schedule improved quality.

Artifacts: [`sentinel_distillation_v3`](../../../../artifacts/training_runs/faithsr3x/sentinel_distillation_v3/).

References: Conde et al., *Swin2SR* (2022); Hinton et al., *Distilling the Knowledge in a Neural Network* (2015).
