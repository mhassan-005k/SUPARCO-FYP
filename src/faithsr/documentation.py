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
    rows = _parameter_rows(model)
    activations = _activation_counts(model)
    layer_table = "\n".join(f"| `{name}` | {kind} | {parameters:,} |" for name, kind, parameters in rows)
    activation_table = "\n".join(f"| {name} | {count} | " + ({"GELU": "Smooth feature transformation; preserves small negative evidence.", "Sigmoid": "Bounds gates, severities, or uncertainty to [0,1].", "Softmax": "Makes observation-kernel coefficients positive and unit-sum."}.get(name, "Configured nonlinear transform.")) + " |" for name, count in activations.items()) or "| None | 0 | Linear ablation control. |"
    optional_math = []
    if da:
        optional_math.append(r"\(\bar h_i=(1+\gamma_i(z))\odot h_i+\beta_i(z)\) (degradation-conditioned FiLM).")
    if observation:
        optional_math.append(r"\(k_z=\operatorname{softmax}(W_kz),\;\tilde x=\downarrow_3(k_z*\hat y_0),\;\hat y=\hat y_0+\alpha(z)\operatorname{Bicubic}_3(x-\tilde x),\;0\le\alpha\le0.25\).")
    if uncertainty:
        optional_math.append(r"\(u=\operatorname{Bilinear}_3(\sigma(C_{u2}(\operatorname{GELU}(C_{u1}(f)))))\).")
    card = fr"""# Experiment Model Card: {experiment_id}

## Purpose and lineage

- Why this experiment was run: {rationale}
- Previous experiment/connection: {previous_experiment}
- Scale: exact native 3x
- Trainable parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}
- Degradation adaptation: {da}
- Learned observation consistency: {observation}
- Spatial uncertainty: {uncertainty}

## Mathematical forward function

\[
f_0=C_{{3\times3}}(x),\qquad
h_i=P_{{2,i}}(\operatorname{{GELU}}(D_i(\operatorname{{GELU}}(P_{{1,i}}(f_{{i-1}}))))),
\]

\[
f_i=f_{{i-1}}+s_i h_i\odot\sigma(G_i(h_i)),\qquad
\hat y_0=\operatorname{{Bicubic}}_3(x)+\operatorname{{PixelShuffle}}_3(C_r(f)).
\]

""" + ("\n\n".join(optional_math) if optional_math else "This control variant intentionally omits degradation conditioning, learned observation correction, and uncertainty.") + fr"""

The training objective and degradation equations are frozen in `docs/research/model-and-experiment-methodology.md` and instantiated by the exact configuration below.

## Architecture figure

```mermaid
flowchart LR
    X["LR input"] --> S["3x3 stem"]
    X --> E["Degradation encoder: {da or observation}"]
    S --> B["Six residual blocks; FiLM={da}"]
    E --> B
    B --> F["Multi-depth fusion"]
    F --> P["Conv + PixelShuffle x3"]
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

PixelShuffle is a deterministic tensor rearrangement, not an activation. For RGB x3, the preceding convolution predicts 3x3²=27 sub-pixel channels.

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
