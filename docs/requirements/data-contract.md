# Input, Output, and Metadata Contract

Status: Current implementation contract - 8-bit TIFF path; superseded as a complete target by SUPARCO SRS v1.2

Last updated: 2026-09-18

> **Scope notice:** This file describes the currently implemented 8-bit RGB path. SUPARCO SRS v1.2 additionally requires 10/12-bit, panchromatic, and multispectral support. See `srs-v1.2-compliance-audit.md` for the current gap assessment and `srs-v1.2-implementation-roadmap.md` for the closure plan.

## Terminology that must not be mixed

| Term | Meaning | Example |
|---|---|---|
| 8-bit | Numeric depth of each band value | `uint8`, usually 0-255 |
| 16-bit | Numeric storage with greater radiometric range | `uint16`, usually 0-65535 |
| 8-band | Eight spectral values/channels per pixel | Eight wavelength bands |
| Metadata | Information describing the raster | CRS, transform, GSD, no-data, tags |

An 8-bit image does not carry eight values as metadata. A three-band 8-bit RGB pixel carries three numeric values, and each value is stored using eight bits. An eight-band image carries eight values per pixel; those values may themselves be 8-bit, 16-bit, or another data type.

## Confirmed requirements

- Input container will be `.tiff` (the reader may also accept `.tif` as an alias).
- Input pixel values will be 8-bit and must not be silently changed to another output bit depth.
- Spatial scaling must be at least 3x in width and height. Native 3x and native 4x outputs are both valid candidates.
- All available TIFF/GeoTIFF metadata must be carried into the output where the field remains valid.
- Metadata fields that depend on raster dimensions or pixel spacing must be recalculated for the 3x raster rather than copied unchanged.
- The reconstruction must prioritize data fidelity and minimum unsupported detail over perceptual sharpness.

## Details still requiring written confirmation

1. What is the exact band count and band order?
2. Is the file a georeferenced GeoTIFF or an ordinary TIFF?
3. Are pixel values raw digital numbers or a display-scaled/processed product?
4. Which no-data value and valid-data mask are used?
5. Must every input band be super-resolved, or only selected RGB/panchromatic bands?
6. What hardware must satisfy the sub-one-second latency requirement?
7. Does the one-second limit cover model inference only or complete file read, processing, stitching, and write time?

The working RGB contract below is implemented and covered by automated round-trip tests. These unanswered items remain hard gates for restricted SUPARCO validation; the software does not infer or invent them from an uploaded file.

## Training versus application input

The application does **not** blur, downsample, compress or otherwise synthetically degrade the uploaded TIFF. Its inference path is:

`observed TIFF -> validate bands/dtype -> normalize values -> model x3/x4 -> clip and restore uint8 -> write metadata-preserving TIFF`

Synthetic degradation exists only in supervised training and paired public evaluation. There, a known higher-resolution training tile is degraded to create an LR input because public Sentinel datasets do not supply a true same-scene x4 target. Consequently, synthetic test scores estimate reconstruction under the registered degradation family; they do not guarantee the same quality on an arbitrary operational sensor image.

The current application model contract is three-band RGB. A 13-band EuroSAT TIFF cannot be passed to that checkpoint and have ten bands silently reattached. The forthcoming multispectral experiment must declare either (a) a band-aware model that reconstructs every selected band, or (b) an RGB-only output plus an explicit untouched-band policy. Output metadata may describe generated data, but metadata preservation cannot manufacture spectral predictions the network never made.

## Draft input contract

- Container: `.tiff`; `.tif` may be accepted as an equivalent extension.
- Storage depth: 8-bit (`uint8`) input and 8-bit (`uint8`) output.
- Current implemented application path: native 3x in both spatial dimensions.
- Product acceptance range: scale factor `s >= 3`; the chosen model's native scale must be declared in the audit.
- Required tile size: 512x512 input pixels; output tile is 1536x1536 pixels.
- MVP band schema: three-band RGB until SUPARCO confirms a multispectral schema.
- Internal model representation: normalized floating point for inference; values are clipped, rounded, and restored to 8-bit only at the output boundary.
- Large rasters: processed with overlapping windows and weighted stitching.

## Draft output contract

The output shall preserve or correctly transform:

- coordinate reference system (CRS);
- affine/geotransform and geographic extent;
- band count, order, names, descriptions, and color interpretation;
- original data type and radiometric scale/offset where scientifically valid;
- no-data value and valid-data mask;
- dataset and per-band tags;
- acquisition/sensor tags supplied in the input;
- compression/profile choices where compatible with the enlarged raster.

Only scale-dependent fields change. For the selected declared scale factor `s`:

- `width_out = width_in * s`
- `height_out = height_in * s`
- horizontal and vertical pixel spacing are divided by `s`
- geographic bounds and CRS remain unchanged

The new pixel spacing is a nominal, model-derived GSD. It must not be represented as equivalent to newly captured sensor measurements.

## Mandatory metadata tests

1. Create a synthetic GeoTIFF with a known CRS, transform, tags, no-data value, mask, and band descriptions.
2. Run the complete 3x inference pipeline.
3. Reopen the output with Rasterio.
4. Assert unchanged CRS and bounds.
5. Assert output width/height match the declared scale and pixel spacing is divided by that scale. The current native-3x regression fixture specifically checks 3x and one-third spacing.
6. Assert preserved data type, band schema, tags, no-data value, and mask semantics.
7. Compare tiled and non-tiled output on a small fixture.

## References

- OGC GeoTIFF 1.1: https://www.ogc.org/standards/geotiff/
- OGC GeoTIFF specification: https://docs.ogc.org/is/19-008r4/19-008r4.html
- Sentinel-2 products: https://sentiwiki.copernicus.eu/web/s2-products
