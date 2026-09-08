# Dataset Strategy for 3x Super-Resolution

Status: Working decision

Last updated: 2026-09-08

## Recommended role of each dataset

| Dataset | Role | Limitation |
|---|---|---|
| UC Merced | Fast smoke tests and continuity with existing experiments | Small 8-bit RGB scene-classification dataset; not a real LR-HR pair dataset |
| Sentinel-2 Level-2A | Main public-domain training and scientific evaluation source | Native data is multi-resolution and does not provide a true same-band 3x HR counterpart |
| SUPARCO imagery | Final on-site validation and domain adaptation | Restricted access and limited working slots |

## Sentinel-2 facts relevant to this project

Sentinel-2 MSI provides 13 bands:

- 4 bands at 10 m: B2, B3, B4, B8;
- 6 bands at 20 m: B5, B6, B7, B8A, B11, B12;
- 3 bands at 60 m: B1, B9, B10.

Sentinel-2 reflectance products use 16-bit integer storage with quantification and radiometric-offset metadata. They must not be silently converted to 8-bit RGB for scientific experiments. Quality/classification layers may use different encodings.

Sources:

- ESA Sentinel-2 facts: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures
- Sentinel-2 Product Specification: https://sentinels.copernicus.eu/documents/d/sentinel/sentinel-2-products-specification-document-15_1
- Sentinel-2 processing and quantification: https://sentiwiki.copernicus.eu/web/s2-processing

## Proposed public-data experiment

For a controlled 3x SISR experiment:

1. Use cloud-filtered Sentinel-2 Level-2A 10 m bands as the HR target.
2. For the RGB experiment, use B4/B3/B2; optionally include B8 as a fourth-band experiment.
3. Convert digital numbers to physically meaningful surface reflectance using product quantification and offset metadata.
4. Generate synthetic 30 m LR observations from the 10 m target with an MTF-inspired blur, 3x decimation, sensor noise, and radiometric quantization.
5. Train the model to reconstruct the 10 m target from the simulated 30 m input.
6. Preserve geographic region, acquisition date, and tile identifiers in the manifest.
7. Split by geographic tile and acquisition, never by random patches from the same source image.

This experiment is scientifically controlled but still synthetic. It cannot by itself prove performance on real unknown SUPARCO degradations.

## Final validation path

1. Pretrain and ablate on public datasets.
2. Freeze the public-data baseline before SUPARCO testing.
3. Evaluate the frozen model on restricted on-site imagery.
4. If permitted, fine-tune only compact adapters/degradation modules on-site.
5. Compare frozen and adapted models on a separate on-site holdout set.
6. Export only artifacts permitted by the SUPARCO data-use agreement.

## Data quality filters

- exclude or mask cloud, cloud-shadow, snow, saturated, defective, and no-data pixels;
- retain acquisition date, processing baseline, solar/viewing geometry, band identifiers, CRS, and transform;
- avoid mixing incompatible radiometric products without normalization;
- record every transformation in a machine-readable manifest;
- keep source imagery and derived LR data in separate immutable locations.
