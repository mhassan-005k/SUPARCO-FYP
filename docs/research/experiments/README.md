# Experiment Dossiers

Yeh folder research paper ka experiment-wise source of truth hai. Har experiment ka apna folder aur `README.md` hai. Central [`experiment-log.md`](../experiment-log.md) chronological audit trail hai; yahan har run ka sawal, model function, layers, activations, result, failure aur next decision aik jagah milta hai.

## Mathematical documentation rule

“Human-readable function” ka matlab pseudo-code nahi. Har paper-facing function proper display equation mein likha jayega, for example:

$$
\mathbf{x}\in[0,1]^{B\times C\times H\times W},
\qquad
\hat{\mathbf{y}}\in[0,1]^{B\times C\times 4H\times 4W}.
$$

Markdown mein inline mathematics ke liye `$...$` aur display mathematics ke liye `$$...$$` use honge. Is tarah Pandoc/LaTeX workflow equation ko Word `.docx` mein editable Word Equation (OMML) bana sakta hai; formula ko backticks mein pseudo-code bana kar paper source nahi rakha jayega.

Har equation ke foran baad yeh cheezen lazmi hain:

- har symbol ka plain-language meaning aur tensor shape;
- equation ka code class/function mapping;
- layer-by-layer kernel, channels, stride, padding and parameter count;
- har activation function ka formula, output range, exact location and selection reason;
- loss ke har term, weight and optimization effect ki explanation;
- previous experiment se connection aur is function ko test karne ki wajah.

Canonical reusable format: [`experiment-template.md`](../experiment-template.md).

## Experiment index

| Dossier | Purpose | Final state |
|---|---|---|
| [`legacy-faithsr-v01-uc-merced`](legacy-faithsr-v01-uc-merced/) | Original lightweight x4 ablation on UC Merced | Historical baseline; DA gain small |
| [`eurosat-rgb-native-x3`](eurosat-rgb-native-x3/) | First native-x3 public-data training and matched benchmark | Positive synthetic-degradation baseline |
| [`sentinel-delhi-ablation`](sentinel-delhi-ablation/) | Geographic x3 component ablation | DA supported; other components mixed |
| [`sentinel-distillation-x3`](sentinel-distillation-x3/) | Frozen generic Swin2SR teacher to x3 student | Distillation neutral |
| [`sentinel-native-l2a-x3`](sentinel-native-l2a-x3/) | Native L2A scene-separated x3 validation | Mixed/negative; uncertainty rejected |
| [`swin2sr-full-finetune-x3-comparison`](swin2sr-full-finetune-x3-comparison/) | Full pretrained personalization under matched x3 output | Fidelity positive; model large |
| [`swin2sr-native-x4-diagnostic`](swin2sr-native-x4-diagnostic/) | Native x4 quality, stress and latency diagnosis | Strong quality; latency failed |
| [`faithsr-lite-x4-personalized-ablation`](faithsr-lite-x4-personalized-ablation/) | Fast student, distillation and compression mechanisms | Latency passed; quality claim failed |
| [`eurosat-multispectral-x4`](eurosat-multispectral-x4/) | Linked Sentinel-2 TIFF band/domain experiment | Planned; file audit required |
| [`hybrid-x4-capacity-context`](hybrid-x4-capacity-context/) | Medium single-pass CNN/context hybrid | Planned; latency and quality gates drafted |
| [`deployment-tiff-latency`](deployment-tiff-latency/) | Complete TIFF read-to-write timing | Sub-second with suitable output mode |
| [`precision-and-export`](precision-and-export/) | ONNX validity and FP16 fidelity/speed | ONNX verified; FP16 optional |
| [`downstream-landcover`](downstream-landcover/) | Frozen semantic probe | Modest positive utility evidence |
| [`qualitative-backfill`](qualitative-backfill/) | Deterministic image-level evidence | Worst/median/best cases retained |

## Required sections for every future experiment

1. Research question and why the run was needed.
2. Connection to the previous accepted result.
3. Dataset, split, seed, preprocessing and leakage controls.
4. Human-readable forward function and tensor shapes.
5. Layer-by-layer architecture and activation rationale.
6. Loss function in words and symbols.
7. Hardware, training budget and checkpoint rule.
8. Quantitative metrics with paired confidence intervals where possible.
9. Saved result images, including a failure case.
10. TIFF/metadata audit when the experiment is deployable.
11. Honest outcome: accepted, mixed, negative, failed, or superseded.
12. References and exact artifact paths.
