from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_an_uploaded_file():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "FaithSR-3X Satellite TIFF Super-Resolution"
