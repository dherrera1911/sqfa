"""Functions for working with colors and colormaps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize


def get_normalized_color_map(color_map, values, color_limits=None):
    """
    Get the color map and normalizer for the given values.

    Parameters
    ----------
    color_map : str or matplotlib.colors.Colormap
        The colormap to use.
    values : list or numpy.ndarray or torch.Tensor
        The value of each class to be used for the color scale
    color_limits : list, optional
        The minimum and maximum values for the color scale. If not provided,
        the minimum and maximum values of `values` will be used.

    Returns
    -------
    color_map : matplotlib.colors.Colormap
        The colormap to use.
    color_normalizer : matplotlib.colors.Normalize
        The normalizer for the colormap.
    """
    values = np.array(values)
    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)
    if color_limits is None:
        color_limits = [np.min(values), np.max(values)]
    color_normalizer = Normalize(vmin=color_limits[0], vmax=color_limits[1])
    return color_map, color_normalizer


def get_class_rgba(color_map, values, color_limits=None):
    """
    Get the RGBA color for each class based on the colormap and the values.

    Parameters
    ----------
    color_map : str or matplotlib.colors.Colormap
        The colormap to use.
    values : list or numpy.ndarray or torch.Tensor
        The value of each class to be used for the color scale
    color_limits : list, optional
        The minimum and maximum values for the color scale. If not provided,
        the minimum and maximum values of `values` will be used.

    Returns
    -------
    colors : numpy.ndarray (n_classes, 4)
        The color for each class in RGBA format.
    """
    color_map, color_normalizer = get_normalized_color_map(
        color_map, values, color_limits
    )
    class_colors = color_map(color_normalizer(values))
    return class_colors


def draw_color_bar(colormap, limits, fig, title=None):
    """
    Draw a color bar for the given colormap and limits.

    Parameters
    ----------
    colormap : str or matplotlib.colors.Colormap
        The colormap to use.
    limits : list
        The minimum and maximum values for the color scale.
    fig : matplotlib.figure.Figure
        The figure to draw the color bar on.

    Returns
    -------
    color_bar: ColorbarBase
        The created color bar.
    """
    color_map, color_normalizer = get_normalized_color_map(colormap, [0, 1], limits)

    # Create an axis for the color bar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])

    # Create the color bar
    color_bar = ColorbarBase(
        cbar_ax, cmap=plt.get_cmap(colormap), norm=color_normalizer
    )

    # Add title if provided
    if title is not None:
        color_bar.set_label(title)
    return color_bar
