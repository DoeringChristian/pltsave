"""
pltsave - A comprehensive matplotlib figure serialization library.

This library allows you to save matplotlib figures by serializing their internal
data structure, and then reload them for interactive plotting. It supports all
major matplotlib features including 2D plots, 3D plots, images, annotations,
legends, colorbars, and more.
"""

from .serializer import save_figure
from .deserializer import load_figure
from .comparator import compare_figures

__version__ = "0.1.0"
__all__ = ["save_figure", "load_figure", "compare_figures"]
