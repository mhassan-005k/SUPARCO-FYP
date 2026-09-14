from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import torch
from torch import nn
import yaml


def _parameter_rows(model: nn.Module) -> list[tuple[str, str, int]]:
    rows = []
    for name, module in model.named_children():
        rows.append((name, module.__class__.__name__, sum(parameter.numel() for parameter in module.parameters())))
    return rows


def _activation_counts(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, (nn.GELU, nn.Sigmoid, nn.Softmax, nn.ReLU, nn.LeakyReLU, nn.SiLU)):
            name = module.__class__.__name__
            counts[name] = counts.get(name, 0) + 1
    return counts


def write_model_card(path: str | Path, model: nn.Module, config: dict[str, Any], experiment_id: str, rationale: str, previous_experiment: str) -> Path:
    """Write the mandatory per-run architecture/mathematics/rationale Markdown card."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    model_config = getattr(model, "config", None)
    da = bool(getattr(model_config, "use_degradation_adaptation", False))
    observation = bool(getattr(model_config, "use_observation_consistency", False))
    uncertainty = bool(getattr(model_config, "use_uncertainty", False))
    compression = bool(getattr(model_config, "use_compression_refinement", False))
    scale = int(getattr(model_config, "scale", 3))
    num_blocks = int(getattr(model_config, "num_blocks", 0))
    rows = _parameter_rows(model)
    activations = _activation_counts(model)
    layer_table = "\n".join(f"| `{name}` | {kind} | {parameters:,} |" for name, kind, parameters in rows)
    activation_table = "\n".join(f"| {name} | {count} | " + ({"GELU": "Smooth feature transformation; preserves small negative evidence.", "Sigmoid": "Bounds gates, severities, or uncertainty to [0,1].", "Softmax": "Makes observation-kernel coefficients positive and unit-sum."}.get(name, "Configured nonlinear transform.")) + " |" for name, count in activations.items()) or "| None | 0 | Linear ablation control. |"
    optional_math = []
    if da:
        optional_math.append(r"\(\bar h_i=(1+\gamma_i(z))\odot h_i+\beta_i(z)\) (degradation-conditioned FiLM).")
    if observation:
        optional_math.append(fr"\(k_z=\operatorname{{softmax}}(W_kz),\;\tilde x=\downarrow_{scale}(k_z*\hat y_0),\;\hat y=\hat y_0+\alpha(z)\operatorname{{Bicubic}}_{scale}(x-\tilde x),\;0\le\alpha\le0.25\).")
    if compression:
        optional_math.append(r"\(f_c=f+s_c\,c(z)\,\sigma(W_cz)\odot R_c(f)\) (compression-gated LR-space refinement).")
    if uncertainty:
        optional_math.append(r"\(u=\operatorname{Bilinear}_3(\sigma(C_{u2}(\operatorname{GELU}(C_{u1}(f)))))\).")
    card = fr"""# Experiment Model Card: {experiment_id}

## Purpose and lineage

- Why this experiment was run: {rationale}
- Previous experiment/connection: {previous_experiment}
- Scale: exact native {scale}x
- Trainable parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}
- Degradation adaptation: {da}
- Learned observation consistency: {observation}
- Spatial uncertainty: {uncertainty}
- Compression-gated refinement: {compression}

## Mathematical forward function

\[
f_0=C_{{3\times3}}(x),\qquad
h_i=P_{{2,i}}(\operatorname{{GELU}}(D_i(\operatorname{{GELU}}(P_{{1,i}}(f_{{i-1}}))))),
\]

\[
f_i=f_{{i-1}}+s_i h_i\odot\sigma(G_i(h_i)),\qquad
\hat y_0=\operatorname{{Bicubic}}_{scale}(x)+\operatorname{{PixelShuffle}}_{scale}(C_r(f)).
\]

""" + ("\n\n".join(optional_math) if optional_math else "This control variant intentionally omits degradation conditioning, learned observation correction, and uncertainty.") + fr"""

The training objective and degradation equations are frozen in `docs/research/model-and-experiment-methodology.md` and instantiated by the exact configuration below.

## Architecture figure

```mermaid
flowchart LR
    X["LR input"] --> S["3x3 stem"]
    X --> E["Degradation encoder: {da or observation or compression}"]
    S --> B["{num_blocks} residual blocks; FiLM={da}"]
    E --> B
    B --> F["Multi-depth fusion"]
    F --> P["Conv + PixelShuffle x{scale}"]
    X --> C["Bicubic base"]
    P --> Y0["Initial SR"]
    C --> Y0
    Y0 --> O["Learned observation correction={observation}"]
    O --> Y["Final SR"]
    F --> U["Uncertainty head={uncertainty}"]
```

## Top-level layers and parameters

| Name | Layer/module | Parameters |
|---|---|---:|
{layer_table}

## Activations and rationale

| Activation | Module count | Why used |
|---|---:|---|
{activation_table}

PixelShuffle is a deterministic tensor rearrangement, not an activation. For RGB x{scale}, the preceding convolution predicts 3x{scale}²={3 * scale**2} sub-pixel channels.

## Exact configuration

```yaml
{yaml.safe_dump(config, sort_keys=False).rstrip()}
```

## References

- Swin2SR: https://arxiv.org/abs/2209.11345
- GELU: https://arxiv.org/abs/1606.08415
- FiLM: https://arxiv.org/abs/1709.07871
- PixelShuffle/sub-pixel convolution: https://arxiv.org/abs/1609.05158
- Canonical project methodology: `docs/research/model-and-experiment-methodology.md`
"""
    target.write_text(card, encoding="utf-8")
    return target


def write_hybrid_x4_model_card(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any],
    experiment_id: str,
    rationale: str,
    previous_experiment: str,
) -> Path:
    """Write a Word-friendly, variant-specific card for the hybrid x4 study."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    rows = _parameter_rows(model)
    activations = _activation_counts(model)
    layer_table = "\n".join(
        f"| `{name}` | {kind} | {parameters:,} |" for name, kind, parameters in rows
    )
    activation_table = "\n".join(
        f"| {name} | {count} | "
        + {
            "GELU": "Smooth nonlinear feature transform; keeps small negative evidence instead of hard-clipping it.",
            "Sigmoid": "Bounds fusion or correction gates between 0 and 1.",
        }.get(name, "Configured nonlinear transform.")
        + " |"
        for name, count in activations.items()
    ) or "| None | 0 | Linear control. |"
    variant_math = {
        "medium_base": r"""The control stops at the initial reconstruction:

$$
\widehat{Y} = \widehat{Y}_0.
$$""",
        "context_fusion": r"""The context branch observes the same real LR feature map at several spatial scales:

$$
C = \operatorname{Mix}\!\left(\operatorname{Concat}\left(B_1(F),\ldots,B_L(F)\right)\right),
$$

$$
G = \operatorname{Sigmoid}\!\left(C_{1\times1}\left(\operatorname{GELU}\left(C_{1\times1}([F,C])\right)\right)\right),
$$

$$
F_{\mathrm{fused}} = F + G \odot C.
$$

The gate lets the network use broad context only where the image supports it.""",
        "residual_cascade": r"""The cascade does not degrade the application input. It only re-observes its own first SR estimate:

$$
\widetilde{X}_0 = \operatorname{AreaDownsample}_{4}(\widehat{Y}_0),
\qquad E = X - \widetilde{X}_0.
$$

The second-stage expert predicts a residual from the real LR input and this consistency error:

$$
\Delta Y = R_{\theta}([X,E]),
\qquad A = 0.15\,\operatorname{Sigmoid}(G_{\theta}(R_{\theta}([X,E]))),
$$

$$
\widehat{Y} = \widehat{Y}_0 + A \odot \Delta Y.
$$

The maximum correction strength of 0.15 is a fidelity guardrail against uncontrolled detail injection.""",
    }[cfg.variant]
    route = {
        "medium_base": 'F --> R["x4 reconstruction"]',
        "context_fusion": 'F --> C["Multi-scale context"]\n    C --> G["Sigmoid fusion gate"]\n    G --> R["x4 reconstruction"]',
        "residual_cascade": 'F --> R["Initial x4 reconstruction"]\n    R --> E["Re-observe output; LR error"]\n    X --> E\n    E --> Q["Bounded residual expert"]\n    Q --> Y["Final x4 output"]',
    }[cfg.variant]
    card = fr"""# Experiment Model Card: {experiment_id}

## Purpose and lineage

- Why this experiment was run: {rationale}
- Previous experiment/connection: {previous_experiment}
- Variant: `{cfg.variant}`
- Scale: native 4x (satisfies the minimum 3x requirement)
- Input/output: matching image bands; experiments use 8-bit, three-band TIFF imagery
- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}

## Application-input rule

The deployed app passes the user's normalized TIFF pixels directly to the model. Synthetic degradation is a supervised training-data construction method only; it is not an inference prerequisite. Original TIFF metadata is handled by the inference pipeline, not encoded into neural pixels.

## Common mathematical model

Let the input be a batch of normalized LR images with shape `batch x bands x height x width`:

$$
X \in [0,1]^{{B \times C \times H \times W}}.
$$

The common local backbone is:

$$
F_0 = C_{{3\times3}}(X),
$$

$$
F_i = F_{{i-1}} + s_i\,C_{{1\times1}}\!\left(DW_{{3\times3}}\!\left(\operatorname{{GELU}}\!\left(C_{{1\times1}}\!\left(\operatorname{{GELU}}\!\left(DW_{{3\times3}}(F_{{i-1}})\right)\right)\right)\right)\right),
$$

where `DW` is a depthwise convolution and each learned residual scale starts at 0.1. Multi-depth features are concatenated and fused:

$$
F = F_0 + C_{{3\times3}}\!\left(\operatorname{{GELU}}\!\left(C_{{1\times1}}([F_1,\ldots,F_N])\right)\right).
$$

The first native-x4 estimate is:

$$
\widehat{{Y}}_0 = \operatorname{{Bicubic}}_4(X) + \operatorname{{PixelShuffle}}_4(C_r(F)).
$$

## Variant-specific function

{variant_math}

## Architecture figure

```mermaid
flowchart LR
    X["Real LR TIFF pixels"] --> S["3x3 shallow features"]
    S --> B["{cfg.num_blocks} efficient residual blocks"]
    B --> F["Multi-depth fusion"]
    {route}
    R --> Y["Native x4 output"]
```

## Top-level layers and parameters

| Name | Layer/module | Parameters |
|---|---|---:|
{layer_table}

## Activations and rationale

| Activation | Count | Why used |
|---|---:|---|
{activation_table}

PixelShuffle is a deterministic rearrangement, not an activation. For RGB x4, its preceding convolution predicts `3 x 4² = 48` sub-pixel channels.

## Exact configuration

```yaml
{yaml.safe_dump(config, sort_keys=False).rstrip()}
```

## Code mapping

- Common block: `EfficientResidualBlock` in `src/faithsr/hybrid_x4.py`
- Context hypothesis: `MultiScaleContextBranch`
- Cascade hypothesis: `ResidualConsistencyExpert`
- End-to-end model: `HybridFaithSRX4`
- Loss implementation: `CompressionAwareObjective` in `src/faithsr/losses.py`

## References

- SAFMN context motivation: https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf
- EFDN efficient residual processing: https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Wang_Edge-Enhanced_Feature_Distillation_Network_for_Efficient_Super-Resolution_CVPRW_2022_paper.html
- PixelShuffle/sub-pixel convolution: https://arxiv.org/abs/1609.05158
- GELU: https://arxiv.org/abs/1606.08415
- Previous experiment: {previous_experiment}
"""
    target.write_text(card, encoding="utf-8")
    return target


def append_run_results(card_path: str | Path, metrics_path: str | Path, duration_seconds: float | None = None) -> Path:
    """Idempotently append exact best/final rows to a generated run card."""

    card = Path(card_path)
    with Path(metrics_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return card
    best = max(rows, key=lambda row: float(row["psnr"]))
    final = rows[-1]
    metric_names = [name for name in best if name not in {"epoch", "learning_rate"} and best[name] not in {"", None}]
    header = "| Checkpoint row | Epoch | " + " | ".join(metric_names) + " |"
    separator = "|---|---:|" + "---:|" * len(metric_names)
    def result_row(label: str, row: dict[str, str]) -> str:
        return f"| {label} | {row['epoch']} | " + " | ".join(row[name] for name in metric_names) + " |"
    duration = "Not recorded" if duration_seconds is None else f"{duration_seconds:.2f} seconds"
    section = "\n".join([
        "<!-- GENERATED_RESULTS_START -->",
        "## Measured training result",
        "",
        "Checkpoint selection rule: maximum validation PSNR. These values come directly from `metrics.csv`.",
        f"Training duration: {duration}.",
        "",
        header, separator, result_row("Best", best), result_row("Final", final),
        "<!-- GENERATED_RESULTS_END -->",
    ])
    text = card.read_text(encoding="utf-8")
    if "<!-- GENERATED_RESULTS_START -->" in text:
        text = text.split("<!-- GENERATED_RESULTS_START -->", 1)[0].rstrip()
    card.write_text(text + "\n\n" + section + "\n", encoding="utf-8")
    return card
