# EuroSAT Multispectral TIFF x4 Domain Experiment

Status: **planned; data audit required before preregistration**

## Why this experiment is next

FaithSR-Lite-X4 proved that a tiny model can meet latency, but the Full compression/distillation design did not improve overall fidelity. The next sensible variable is training-domain and spectral information, not another fashionable block. The linked EuroSAT/Sentinel-2 dataset supplies 27,000 class-organized multispectral TIFFs in addition to 27,000 JPG renderings.

This experiment asks whether broad Sentinel-2 spectral and land-cover diversity can improve the quality-speed frontier when the final evaluation remains scene-separated and fidelity-focused.

## Dataset contract

Expected TIFF path:

`sentinel_2/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/<class>/<image>.tif`

Ten classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River and SeaLake.

Before any training, representative files from every class must be opened and logged. Required fields are SHA-256, shape, dtype, band count/order, min/max/percentiles, CRS, transform, no-data, tags and mask. The official EuroSAT multispectral archive is described as 13-band original-range Sentinel-2 data, but the actual linked files remain the executable source of truth.

## Training input versus app input

Training creates an LR observation from an original EuroSAT tile so that a supervised target exists. The application is different:

`training: original tile -> registered synthetic degradation -> LR input -> model -> original tile target`

`application: uploaded TIFF -> normalize -> model -> x4 TIFF output`

The application never applies the synthetic degradation generator to a user's image.

## Band-aware model decision

The current RGB checkpoints accept exactly three channels. They cannot scientifically process 13 bands and then copy ten untouched channels into an x4 raster. After the file audit, one of these frozen contracts will be selected:

1. **RGB control:** B4/B3/B2 input and output, initialized from the current Lite/Swin2SR experiment where compatible.
2. **Selected multispectral model:** explicitly named subset of co-aligned bands; input and output channel counts are identical.
3. **Full multispectral model:** all 13 bands after a documented mixed-resolution alignment policy.

The RGB control is mandatory. The multispectral variant is accepted only if the alignment does not falsely imply that native 20 m or 60 m bands contain captured 10 m detail.

## Human-readable model function

For a selected set of `C` bands, input `x` is **B images × C bands × H rows × W columns**. The model returns **B × C × 4H × 4W**.

Provisional compact function:

`output = clip(Bicubic4(input) + PixelShuffle4(reconstruction(backbone(shallow_features(input)))), valid_range)`

The shallow and reconstruction convolutions change from three channels to `C`; the low-resolution backbone remains the controlled component. If degradation conditioning is tested, it must be a separate matched variant.

## Layer and activation rationale

- **3×3 shallow convolution:** combines neighboring spectral/spatial evidence without immediately discarding resolution.
- **Residual low-resolution blocks:** learn only a correction over the interpolation base and keep computation low.
- **GELU:** smooth non-linearity already used in the verified models; changing it would confound the dataset comparison.
- **PixelShuffle(4):** produces exact native x4 dimensions while most operations remain on the small grid.
- **Bicubic residual path:** anchors low-frequency output to the observation; it is not evidence that x4 pixels are sensor measurements.
- **No GAN/diffusion loss:** avoids optimizing visual realism at the expense of geographic fidelity.

## Required matched comparisons

- bicubic x4;
- current Lite Base RGB checkpoint;
- personalized Swin2SR RGB quality ceiling;
- EuroSAT RGB-pretrained compact model;
- selected multispectral compact model, only if the band contract passes audit.

All models must be evaluated on untouched native Sentinel-2 L2A scene groups in addition to EuroSAT. Metrics are PSNR, SSIM, LPIPS for RGB, false-edge, SAM, ERGAS, re-observation L1, P95 latency, peak VRAM and TIFF audit results. Worst/median/best panels must be selected mechanically.

## Acceptance rule

The new model must retain sub-one-second 512-to-2048 model P95 on recorded hardware and improve the selected fidelity objective without worse false-edge or observation behavior. A gain only on synthetic EuroSAT tiles is not enough; it must transfer to an untouched native-L2A scene test. True physical x4 recovery remains unproven without a registered higher-resolution reference.

## References

- Kaggle dataset: https://www.kaggle.com/datasets/gallo33henrique/sentinel-2-satellite-imagery
- EuroSAT repository: https://github.com/phelber/EuroSAT/blob/master/README.md
- EuroSAT Zenodo record: https://zenodo.org/records/7711810
- Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification* (2019).
