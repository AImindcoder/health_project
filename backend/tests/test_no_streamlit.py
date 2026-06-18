"""Verify the FastAPI backend never imports streamlit."""

import sys


def test_no_streamlit_in_backend():
    assert "streamlit" not in sys.modules, (
        "Streamlit must not be imported in FastAPI context. "
        "Check the import chain from app/main.py for any streamlit references."
    )


def test_no_streamlit_in_guard_comment():
    with open("app/main.py") as f:
        content = f.read()
    assert "NOTE: No streamlit imports allowed in this file" in content
