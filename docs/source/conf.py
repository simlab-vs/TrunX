"""Sphinx configuration for the trunx documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

project = "trunx"
copyright = "2026, TrunX contributors"
author = "TrunX contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_mock_imports = ["jax", "jaxlib"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_theme_options = {
    "sidebar_hide_name": True,
}
