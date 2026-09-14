# ONNX Export and FP32/FP16 Precision

## Research question

Can the custom model be exported at dynamic image sizes, and can FP16 reduce deployment cost without materially changing the required 8-bit TIFF?

## ONNX function

The frozen native-x3 graph maps **one RGB image of any H×W size** to **one RGB image of 3H×3W**. Opset 18 was checked and executed at 64×64 and 67×69 inputs. ONNX Runtime matched PyTorch with maximum absolute float error 0.0000104.

$$
F_{\theta}^{\mathrm{ONNX}}:\mathbb{R}^{1\times3\times H\times W}
\rightarrow\mathbb{R}^{1\times3\times3H\times3W},
$$

$$
\epsilon_{\max}=\max_j\left|F_{\theta}^{\mathrm{PyTorch}}(\mathbf{x})_j
-F_{\theta}^{\mathrm{ONNX}}(\mathbf{x})_j\right|=1.04\times10^{-5}.
$$

Code: `faithsr.export` constructs and verifies the ONNX graph; `FaithSR3X.forward` defines $F_{\theta}$.

## Precision experiment

Only CUDA arithmetic precision changed. Model, input, tiling and uncompressed TIFF output stayed fixed.

| Precision | Model P95 | TIFF P95 | Peak VRAM |
|---|---:|---:|---:|
| FP32 | 53.03 ms | 497.86 ms | 3.20 GB |
| FP16 autocast | **52.14 ms** | **450.99 ms** | **3.07 GB** |

After uint8 conversion, maximum output difference was one digital number and mean absolute difference was 0.0679 DN. Metadata was identical.

## Why FP16, and why not INT8 yet

FP16 uses tensor-core-friendly arithmetic while retaining sufficient range for this normalized model. It is an optional speed mode, not bit-identical math. INT8 is deferred because quantization needs a frozen, representative Sentinel calibration set and full fidelity/false-edge validation.

Artifacts: [`precision_benchmark_v1`](../../../../artifacts/results/faithsr3x/precision_benchmark_v1/) and exported ONNX files under [`artifacts/exports`](../../../../artifacts/exports/).

References: ONNX opset documentation; PyTorch automatic mixed precision documentation.
