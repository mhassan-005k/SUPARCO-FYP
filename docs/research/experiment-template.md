# Paper-facing Experiment Template

## Experiment identity and research question

- Experiment ID, date, Git commit and remote kernel/version.
- Exact hypothesis and why the previous experiment could not answer it.
- Accepted, mixed, negative, failed, superseded, or planned state.

## Dataset and split

- Dataset/version, paths and source licence.
- Input bands, order, dtype, radiometric range and metadata.
- Train/validation/test counts, geographic grouping, seed and leakage controls.
- Exact LR-generation or real-pair protocol. State explicitly that application inference does not run training degradation.

## Mathematical model

Write actual mathematics using `$...$` inline and `$$...$$` for display equations. Example input/output declaration:

$$
\mathbf{x}\in[0,1]^{B\times C\times H\times W},
\qquad
F_{\theta}(\mathbf{x})=\hat{\mathbf{y}}
\in[0,1]^{B\times C\times sH\times sW}.
$$

Define every symbol immediately below the equation. Never leave raw text such as `x \in ...` outside math delimiters, and never substitute pseudo-code for the paper's mathematical model.

## Code-to-equation mapping

| Mathematical term | Code symbol | File and line | Responsibility |
|---|---|---|---|
| $F_{\theta}$ | `ModelClass.forward` | `src/...` | Complete forward mapping |
| $\mathcal{L}$ | `Objective.forward` | `src/...` | Training objective |

Include all modules that are active in the experiment and explicitly list disabled/ablated modules.

## Layer specification

For every layer or repeated block record input/output channels, kernel/window size, stride, padding, groups, block count, normalization, residual/skip connection and parameter count. Explain why this design fits satellite fidelity and latency requirements.

## Activation functions

For each activation give the equation, range and exact use. Examples:

$$
\sigma(u)=\frac{1}{1+e^{-u}},\qquad \sigma(u)\in(0,1),
$$

$$
\operatorname{GELU}(u)=u\,\Phi(u),
$$

$$
\operatorname{Softmax}(u_i)=\frac{e^{u_i}}{\sum_j e^{u_j}}.
$$

Explain why each was selected and what would break or change if another activation were used.

## Loss function

Write the complete weighted objective as one rendered equation, then define every term, coefficient, target and enabled variant. Distinguish ground-truth supervision, observation consistency and teacher supervision.

## Architecture figure and execution path

Include a Mermaid architecture figure and separate training/inference paths. Show whether a teacher is training-only. Record TIFF tiling, blending, dtype restoration and metadata handling when applicable.

## Results and saved images

- Epoch table, best-checkpoint rule and checkpoint SHA-256.
- Dataset and profile metrics with sample counts and confidence intervals.
- Model-only and end-to-end P95 latency, hardware, precision and VRAM.
- Mechanically selected worst/median/best panels and same-name JSON evidence.
- TIFF metadata audit and warnings.

## Interpretation, trade-offs and next decision

State what passed, what failed, plausible explanations versus proven causes, and the single next experiment justified by the evidence. Do not convert a visually attractive sample or tiny statistical effect into a strong contribution claim.
