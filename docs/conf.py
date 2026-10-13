"""Sphinx configuration for the Adaptive Signal Processing documentation."""

import os
import sys

# Compile-free import: ``numba.njit`` becomes a no-op so that autodoc sees
# plain functions (with signatures) and the plot directive runs without JIT
# warm-up. Must be set before importing the package.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.abspath("../src"))

import asp as asp  # noqa: E402

# -- Project information -----------------------------------------------------
project = "Adaptive Signal Processing"
author = "Edson Porto da Silva"
copyright = "2024-2026, Edson Porto da Silva (CC0 1.0)"
release = asp.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"
numfig = True
math_numfig = True
math_eqref_format = "({number})"
nitpicky = False

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {"show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_timeout = 5

# Matplotlib plot directive (figures generated at build time)
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]
plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
from asp import algorithms as alg
plt.rcParams.update({
    'figure.figsize': (6.4, 3.0), 'figure.autolayout': True, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
})
"""

# MathJax macros used throughout the theory pages
mathjax3_config = {
    "tex": {
        "macros": {
            "bm": [r"\boldsymbol{#1}", 1],
            "E": [r"\operatorname{E}\!\left[#1\right]", 1],
            "tr": r"\operatorname{tr}",
            "vx": r"\mathbf{x}",
            "vh": r"\mathbf{h}",
            "vw": r"\mathbf{w}",
            "mR": r"\mathbf{R}",
            "vp": r"\mathbf{p}",
        }
    }
}

# -- HTML output -------------------------------------------------------------
try:
    import furo  # noqa: F401

    html_theme = "furo"
    html_theme_options = {
        "source_repository": "https://github.com/edsonportosilva/AdaptiveSignalProcessing/",
        "source_branch": "main",
        "source_directory": "docs/",
        "light_css_variables": {
            "color-brand-primary": "#1f5fa8",
            "color-brand-content": "#1f5fa8",
        },
        "dark_css_variables": {
            "color-brand-primary": "#7fb2ee",
            "color-brand-content": "#7fb2ee",
        },
    }
except ImportError:  # pragma: no cover - furo is optional
    html_theme = "alabaster"

html_title = "Adaptive Signal Processing"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sphinx = False
