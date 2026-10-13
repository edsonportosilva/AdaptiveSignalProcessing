"""Adaptive Signal Processing: course algorithms and helper utilities.

Submodules
----------
algorithms
    Implementations of the adaptive filtering algorithms (LMS family, RLS,
    Kalman filters, CMA, MLP) and the optimization/estimation tools they rely on.
utils
    Plotting, animation and symbolic-display helpers used by the notebooks
    (requires ``matplotlib``, ``sympy`` and ``IPython``).
"""

from importlib import metadata as _metadata

from . import algorithms

try:
    __version__ = _metadata.version("adaptive-signal-processing")
except _metadata.PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = ["algorithms", "__version__"]
