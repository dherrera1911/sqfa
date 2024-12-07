"""Utilities for plotting data, model parameters and geometry."""

from .colors import draw_color_bar
from .data import scatter_data
from .ellipses import statistics_ellipses

__all__ = [
    "scatter_data",
    "statistics_ellipses",
    "draw_color_bar",
]


def __dir__():
    return __all__
