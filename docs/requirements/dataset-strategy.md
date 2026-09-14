# Dataset Strategy for Minimum-x3 Satellite Super-Resolution

Status: Updated working decision

Last updated: 2026-09-14

## Decision

The linked Kaggle collection `gallo33henrique/sentinel-2-satellite-imagery` is a valid EuroSAT/Sentinel-2 source, not a JPEG-only dataset. Its Data Explorer exposes two parallel branches: 27,000 class-organized JPG files and 27,000 class-organized TIFF files (54,000 files total, version 1, about 2.99 GB). The TIFF branch is the preferred branch for the next public-data experiment.

It does not replace the current native Sentinel-2 L2A scene corpus. EuroSAT tiles are only 64 by 64 pixels and are organized for land-cover classification; they do not provide real LR/HR x4 pairs or a realistic 512-by-512 deployment raster. We will therefore use both Sentinel sources for different jobs.

## Role of each dataset

| Dataset | Role | What it can prove | Main limitation |
|---|---|---|---|
| UC Merced | Historical continuity and fast smoke tests | Code/model sanity on familiar RGB scenes | Small 8-bit RGB classification dataset; no real LR/HR pairs |
| Linked EuroSAT TIFF branch | Multispectral public pretraining, land-cover diversity and downstream probe | Whether Sentinel-2 spectral diversity helps a band-aware model | 64-by-64 tiles; synthetic SR pairs; class split can leak geography unless rebuilt |
| Native Sentinel-2 L2A scenes | Primary 512-scale stress, TIFF pipeline and scene-separated evaluation | Performance on documented, independently acquired public Sentinel scenes | Still no true same-band x4 ground truth |
| SUPARCO imagery | Final on-site validation and permitted adaptation | Performance on the actual operational domain | Restricted and contract details are pending |

## Verified linked-dataset structure

The useful TIFF path shown by Kaggle/EuroSAT tooling is:

`sentinel_2/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/<class>/<image>.tif`

Classes are AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River and SeaLake. The parallel class tree contains display-friendly JPG copies.

Official EuroSAT documentation describes 27,000 georeferenced Sentinel-2 images across ten classes. The multispectral archive contains all 13 Sentinel-2 bands in the original value range, whereas the RGB archive contains JPG images. Before training, the exact linked TIFF files will be sampled and audited for band count, dtype, CRS/tags and value range; dataset-card text alone will not override file inspection.

Sources:

- Kaggle dataset: https://www.kaggle.com/datasets/gallo33henrique/sentinel-2-satellite-imagery
- Official EuroSAT repository: https://github.com/phelber/EuroSAT/blob/master/README.md
- Official EuroSAT Zenodo record: https://zenodo.org/records/7711810

## Sentinel-2 facts relevant to the project

Sentinel-2 MSI provides 13 bands: four at 10 m, six at 20 m and three at 60 m. This matters because a 13-band tile is not equivalent to an RGB image, and bands at different native ground sampling distances must not be treated as if already co-registered at identical physical resolution.

For the linked TIFF experiment we will preserve the original multispectral values and record every resampling decision. We will not silently reduce 13 bands to RGB and later “reattach” ten unprocessed bands. The model must either reconstruct every declared output band under an explicit band-aware configuration or process only a declared subset while marking untouched bands separately.

## Next public-data protocol

1. Pin the Kaggle dataset version and audit representative TIFFs from every class.
2. Build a manifest containing relative path, class, hash, shape, dtype, band count, value range and available geospatial metadata.
3. Create train/validation/test groups using geographic identity where coordinates exist. Class-balanced random image splitting alone is not sufficient.
4. Run an RGB control using B4/B3/B2 so it remains comparable with the current three-channel models.
5. Run a multispectral model only after deciding how the 10 m, 20 m and 60 m bands are aligned and which bands the product must output.
6. Generate synthetic LR observations only for training and paired public evaluation. The application will never degrade a user-uploaded TIFF.
7. Keep the native L2A scene corpus as the 512-by-512 latency, TIFF metadata and out-of-distribution stress benchmark.
8. Compare UC Merced only as a historical baseline, not as final evidence.

## Scientific boundary

Neither EuroSAT nor ordinary Sentinel-2 L2A provides a true same-scene x4 higher-resolution target for all Sentinel-2 bands. Synthetic downsampling teaches an inverse degradation function, but it cannot prove recovery of real 2.5 m ground truth. Final physical-validity claims require accurately registered higher-resolution reference imagery or the restricted SUPARCO evaluation protocol.

## Final validation path

1. Pretrain and ablate on the linked EuroSAT TIFF data plus native Sentinel-2 L2A scenes.
2. Freeze the public checkpoint before SUPARCO testing.
3. Evaluate it on restricted on-site imagery.
4. If permitted, adapt only on a separate adaptation split.
5. Compare frozen and adapted models on an untouched on-site holdout.
6. Export only artifacts permitted by the data-use agreement.

## Data quality filters

- exclude or mask cloud, cloud-shadow, snow, saturated, defective and no-data pixels where quality information exists;
- retain acquisition, band, CRS, transform, dtype, radiometric scaling and source identifiers;
- avoid mixing incompatible radiometric products without explicit normalization;
- record every transformation in a machine-readable manifest;
- keep source imagery and derived LR observations in separate immutable locations.
