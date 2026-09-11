from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import rasterio
import streamlit as st
import torch

from faithsr.checkpoint import load_checkpoint
from faithsr.geotiff import capture_metadata, metadata_summary
from faithsr.inference import process_tiff


st.set_page_config(page_title="FaithSR-3X", layout="wide")
st.title("FaithSR-3X Satellite TIFF Super-Resolution")
st.caption("Data-faithful 3x reconstruction with metadata preservation and uncertainty reporting")
checkpoint = st.text_input("Checkpoint path", "models/checkpoints/faithsr3x/best.pth")
precision = st.selectbox("Inference precision", ["fp32", "fp16"] if torch.cuda.is_available() else ["fp32"])
compression = st.selectbox("Output compression", ["input", "deflate", "lzw", "zstd", "none"])
upload = st.file_uploader("Upload an 8-bit TIFF", type=["tif", "tiff"])
if upload:
    session = Path(tempfile.mkdtemp(prefix="faithsr_")); source = session / upload.name; source.write_bytes(upload.getbuffer())
    try:
        with rasterio.open(source) as dataset:
            summary = metadata_summary(capture_metadata(dataset, include_mask=False))
            indexes = list(range(1, min(3, dataset.count) + 1))
            preview_height = min(512, dataset.height)
            preview_width = min(512, dataset.width)
            preview = dataset.read(indexes, out_shape=(len(indexes), preview_height, preview_width))
        if preview.shape[0] == 1: preview = np.repeat(preview, 3, axis=0)
        if preview.shape[0] == 2: preview = np.concatenate([preview, preview[-1:]], axis=0)
        st.subheader("Input metadata"); st.json(summary)
        st.image(preview[:3].transpose(1, 2, 0), caption="Original 8-bit TIFF", clamp=True)
        if st.button("Run 3x super-resolution", type="primary"):
            if not Path(checkpoint).exists(): st.error("Checkpoint not found")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"; model, state = load_checkpoint(checkpoint, device); output = session / f"{source.stem}_faithsr3x.tiff"
                progress = st.progress(0, text="Preparing tiled inference...")
                def report_progress(completed: int, total: int) -> None:
                    progress.progress(completed / total, text=f"Processing tile {completed} of {total}")
                with st.spinner("Processing tiles and preserving metadata..."):
                    audit = process_tiff(source, output, model, device, model_version=f"epoch-{state.get('epoch', 'unknown')}", compression=compression, precision=precision, progress_callback=report_progress)
                progress.progress(1.0, text="Inference and metadata-preserving write complete")
                st.success("3x output complete"); st.json(audit)
                with rasterio.open(output) as dataset:
                    indexes = list(range(1, min(3, dataset.count) + 1))
                    sr_preview = dataset.read(indexes, out_shape=(len(indexes), min(768, dataset.height), min(768, dataset.width)))
                if sr_preview.shape[0] == 1: sr_preview = np.repeat(sr_preview, 3, axis=0)
                if sr_preview.shape[0] == 2: sr_preview = np.concatenate([sr_preview, sr_preview[-1:]], axis=0)
                original_rgb = preview[:3].transpose(1, 2, 0)
                bicubic = np.asarray(Image.fromarray(original_rgb).resize((original_rgb.shape[1] * 3, original_rgb.shape[0] * 3), Image.Resampling.BICUBIC))
                columns = st.columns(3)
                columns[0].image(original_rgb, caption="Original")
                columns[1].image(bicubic, caption="Bicubic 3x")
                columns[2].image(sr_preview[:3].transpose(1, 2, 0), caption="FaithSR-3X")
                if audit.get("uncertainty"):
                    uncertainty_path = Path(audit["uncertainty"]["artifact"])
                    with rasterio.open(uncertainty_path) as dataset:
                        uncertainty_preview = dataset.read(1, out_shape=(min(768, dataset.height), min(768, dataset.width)))
                    st.image(uncertainty_preview, caption="Uncertainty/risk map (brighter = higher risk)", clamp=True)
                    st.download_button("Download uncertainty GeoTIFF", uncertainty_path.read_bytes(), uncertainty_path.name, "image/tiff")
                st.download_button("Download output TIFF", output.read_bytes(), output.name, "image/tiff")
                audit_path = output.with_suffix(output.suffix + ".audit.json"); st.download_button("Download audit JSON", audit_path.read_bytes(), audit_path.name, "application/json")
    except Exception as error:
        st.error(str(error))
