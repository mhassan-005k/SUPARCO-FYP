# Sentinel held-out qualitative cases

Cases were selected mechanically from all 1,728 held-out per-image rows by FaithSR-minus-bicubic PSNR: minimum, median, and maximum. No image was hand-picked for appearance.

| Case | FaithSR PSNR | Bicubic PSNR | Delta | Figure |
|---|---:|---:|---:|---|
| worst_gain | 26.0819 | 27.7163 | -1.6344 | [worst_gain.png](worst_gain.png) |
| median_gain | 23.4494 | 21.5836 | +1.8658 | [median_gain.png](median_gain.png) |
| best_gain | 28.8438 | 23.2379 | +5.6059 | [best_gain.png](best_gain.png) |

Each PNG shows observed LR, bicubic, FaithSR, held-out HR, absolute error, and uncertainty. Its same-name JSON sidecar contains source path, exact degradation parameters/seed, checkpoint epoch, and per-image metrics.
