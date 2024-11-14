"""
Utilities for plotting data, model parameters and geometry.
"""

from .data import scatter_data
from .ellipses import statistics_ellipses
from .colors import draw_color_bar

__all__ = [
    "scatter_data",
    "statistics_ellipses",
    "draw_color_bar",
]


def __dir__():
    return __all__
