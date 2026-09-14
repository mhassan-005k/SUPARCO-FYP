# EuroSAT RGB Native-x3 Baseline

## Research question

Can the active custom FaithSR architecture learn an exact x3 mapping and pass a 512-block model-only latency check before the stricter native Sentinel experiments?

## Data and model

The run used the public EuroSAT RGB JPEG mirror, 21,000 training images and 6,000 held-out images from AnnualCrop and Forest, seed 42. The 117,224-parameter network used six residual/adaptive blocks and 63×63 HR crops for 30 epochs.

Plain-language function: input has **B RGB images × H × W** with values 0–1. A shallow convolution extracts features; six GELU residual blocks refine them; a 27-channel convolution feeds PixelShuffle(3), producing **B RGB images × 3H × 3W**. A bounded observation correction discourages detail that cannot reproduce the LR observation.

`SR = clip(PixelShuffle3(head(blocks(conv(input)))) + constrained correction, 0, 1)`

GELU is smooth and stable in residual blocks. Sigmoid-bounded heads keep degradation/risk outputs interpretable. PixelShuffle is used because spatial resizing is exact and the heavy computation remains in LR space.

## Result and limitation

Best validation was 30.6966 dB PSNR and 0.84078 SSIM at epoch 25. Matched evaluation was 30.6825 dB versus bicubic 22.7390 dB. Tesla P100 model-only P95 was 54.04 ms for 512→1536.

This large gain is synthetic-degradation evidence on 64×64 JPEG patches. It does not prove real Sentinel GeoTIFF recovery, because JPEG conversion has already discarded multispectral radiometry and original metadata.

## Connection and decision

The run validated code, native x3 shape and speed. It led to geographic Sentinel-2 ablations and complete TIFF benchmarks rather than becoming the final result.

Artifacts: [`eurosat_preliminary_v8`](../../../../artifacts/training_runs/faithsr3x/eurosat_preliminary_v8/) and [`eurosat_matched_v3`](../../../../artifacts/results/faithsr3x/eurosat_matched_v3/).

References: Helber et al., *EuroSAT* (2019); Shi et al. (2016); Wang et al., *SSIM* (2004).
