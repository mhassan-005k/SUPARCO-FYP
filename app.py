from __future__ import annotations

from pathlib import Path
import tempfile

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
upload = st.file_uploader("Upload an 8-bit TIFF", type=["tif", "tiff"])
if upload:
    session = Path(tempfile.mkdtemp(prefix="faithsr_")); source = session / upload.name; source.write_bytes(upload.getbuffer())
    try:
        with rasterio.open(source) as dataset: summary = metadata_summary(capture_metadata(dataset))
        st.subheader("Input metadata"); st.json(summary)
        if st.button("Run 3x super-resolution", type="primary"):
            if not Path(checkpoint).exists(): st.error("Checkpoint not found")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"; model, state = load_checkpoint(checkpoint, device); output = session / f"{source.stem}_faithsr3x.tiff"
                with st.spinner("Processing tiles and preserving metadata..."): audit = process_tiff(source, output, model, device, model_version=f"epoch-{state.get('epoch', 'unknown')}")
                st.success("3x output complete"); st.json(audit)
                st.download_button("Download output TIFF", output.read_bytes(), output.name, "image/tiff")
                audit_path = output.with_suffix(output.suffix + ".audit.json"); st.download_button("Download audit JSON", audit_path.read_bytes(), audit_path.name, "application/json")
    except Exception as error:
        st.error(str(error))
