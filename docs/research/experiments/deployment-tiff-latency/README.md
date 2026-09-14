# End-to-End TIFF Deployment Latency

## Why this experiment was necessary

A fast neural forward pass does not guarantee that an analyst receives a TIFF within one second. These runs measured reading, tiling, inference, disk-backed blending, uint8 conversion, metadata handling and writing.

## Pipeline function

The app does **not** synthetically degrade the uploaded TIFF. It performs:

`uploaded TIFF → validate and normalize → overlapping model tiles → weighted blend → restore uint8 and metadata → output TIFF plus audit`

For overlapping tiles, every output pixel is a weighted average of all tile predictions covering it. Hann weights down-weight unreliable tile borders and prevent hard seams. Disk-backed sum and weight rasters avoid allocating the whole enlarged image in GPU memory.

## v5 and v6 results

v5 used lossless Deflate and had 52.96 ms model-only P95 but 1,266.34 ms end-to-end P95; final writing dominated. v6 held the fixture/model fixed and varied only TIFF compression:

| Compression | End-to-end P95 | Outcome |
|---|---:|---|
| Input/Deflate | 743.43 ms | Pass |
| LZW | 697.34 ms | Pass |
| None | **460.42 ms** | Pass |

The v5/v6 spread shows I/O variance, so this is a measured hardware result, not a universal guarantee. Uncompressed output is fastest but larger; lossless compression is a storage/latency choice.

## Connection

These x3 results established the deployment harness later reused for native-x4 Swin2SR and FaithSR-Lite. In the Lite-x4 run, Full achieved 482.85 ms FP32 and 399.19 ms FP16 end-to-end P95.

Artifacts: [`deployment_benchmark_v5`](../../../../artifacts/results/faithsr3x/deployment_benchmark_v5/) and [`deployment_benchmark_v6`](../../../../artifacts/results/faithsr3x/deployment_benchmark_v6/).

References: Rasterio GeoTIFF documentation; Ngambenjavichaikul et al., *Edge AI Satellite SR Platform* (2025).
