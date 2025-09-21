"""Plot ellipses."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

from ._data_wrangle import statistics_dim_subset
from .colors import get_class_rgba


def single_ellipse(covariance, ax, center=None, color="black"):
    """
    Plot an ellipse with a given center and covariance matrix.

    Parameters
    ----------
    covariance : torch.Tensor
        Covariance matrix of the ellipse. Shape (2, 2).
    ax : matplotlib.axes.Axes
        Axes to plot the ellipse.
    center : torch.Tensor, optional
        Center of the ellipse. Shape (2). The default is [0, 0].
    color : optional
        Color of the ellipse. The default is 'black'.
    """
    if center is None:
        center = torch.zeros(2)
    eig_val, eig_vec = torch.linalg.eigh(covariance)
    # Get the angle of the ellipse main axis
    angle = torch.atan2(eig_vec[1, 0], eig_vec[0, 0])
    # Get the length of the axes
    scale = torch.sqrt(eig_val)
    # Plot the ellipse
    ellipse = patches.Ellipse(
        xy=center,
        width=scale[0] * 4,
        height=scale[1] * 4,
        angle=angle * 180 / np.pi,
        color=color,
    )
    ellipse.set_facecolor("none")
    ellipse.set_linewidth(3)
    ax.add_patch(ellipse)


def statistics_ellipses(
    ellipses,
    centers=None,
    dim_pair=(0, 1),
    ax=None,
    values=None,
    classes_plot=None,
    color_map="viridis",
    legend_type="none",
    **kwargs,
):
    """
    Plot the ellipses of the filter response statistics across classes.

    Parameters
    ----------
    ellipses : torch.Tensor
        Shape of ellipses as SPD matrices. Shape (n_classes, n_dim, n_dim).
    centers : torch.Tensor
        Centers of ellipses. Shape (n_classes, n_dim).
    dim_pair : tuple of int, optional
        Pair of filters to plot. The default is [0, 1].
    ax : matplotlib.axes.Axes, optional
        Axes to plot the ellipses. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color code the ellipses. Each value corresponds to a
        class. Default is [0:n_classes].
    classes_plot : list, optional
        List of classes to plot. The default is all classes.
    color_map : str or matplotlib.colors.Colormap, optional
        Color map to use for the ellipses. The default is 'viridis'.
    legend_type : str, optional
        Type of legend to add: 'none', 'continuous', 'discrete'.
        The default is 'none'.
    **kwargs
        Additional arguments to pass to the legend.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    if centers is None:
        centers = torch.zeros(ellipses.shape[0], ellipses.shape[1])

    if classes_plot is None:
        classes_plot = np.arange(centers.shape[0])


    values = np.arange(len(classes_plot)) if values is None else np.array(values)

    # Get color maps
    if values.ndim == 1:
        if isinstance(color_map, str):
            color_map = plt.get_cmap(color_map)
        class_colors = get_class_rgba(color_map, values)
        col1_vals, col2_vals = values, None
    elif values.ndim == 2 and values.shape[1] == 2:
        col1_vals = values[:, 0]
        col2_vals = values[:, 1]
        class_colors = _rgba_from_two_dims(
            col1_vals, col2_vals,
            cmap=color_map,
        )
    else:
        raise ValueError("`values` must be 1‑D or 2‑D with 2 columns.")

    centers_subset, ellipses_subset = statistics_dim_subset(
      centers, ellipses, dim_pair
    )

    for _, ind in enumerate(classes_plot):
        single_ellipse(
            center=centers_subset[ind],
            covariance=ellipses_subset[ind],
            ax=ax,
            color=class_colors[ind],
        )

    ax.autoscale_view()
    ax.set_xlabel(f"Dimension {dim_pair[0] + 1}")
    ax.set_ylabel(f"Dimension {dim_pair[1] + 1}")

#    if legend_type == "continuous" and values.ndim == 1:
#        color_map, norm = get_normalized_color_map(color_map, values)
#        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
#        sm.set_array([])
#        plt.colorbar(sm, ax=ax, **kwargs)
#
#    elif legend_type == "two" and col2_vals is not None:
#        # 1) orientation colourbar
#        cmap, onorm = get_normalized_color_map(color_map, col1_vals)
#        sm = plt.cm.ScalarMappable(cmap=cmap, norm=onorm)
#        sm.set_array([])
#        plt.colorbar(sm, ax=ax, label="Orientation", **kwargs)
#
#        # 2) frequency legend (alpha patches)
#        uniq_f = np.unique(col2_vals)
#        handles = []
#        for f in uniq_f:
#            alpha = alpha_range[0] + (
#                (f - col2_vals.min()) / (col2_vals.ptp() + 1e-12)
#            ) * (alpha_range[1] - alpha_range[0])
#            patch = patches.Patch(
#                facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, alpha),
#                linewidth=3, label=f"{f:g}"
#            )
#            handles.append(patch)
#        ax.legend(handles=handles, title="Frequency")
#
#    elif legend_type == "discrete":
#        for _, ind in enumerate(classes_plot):
#            ax.scatter([], [], c=[class_colors[ind]], label=values[ind])
#        ax.legend(**kwargs)

    return ax


def _rgba_from_two_dims(dim1_vals, dim2_vals,
                        cmap='hsv', alpha_range=(0.3, 1.0)):
    """
    Map `orient` to RGB using `cmap` and `freq` to alpha.

    orient, freq : 1‑D numpy arrays of equal length
    returns      : (N, 4) float32 RGBA array
    """
    cmap = plt.get_cmap(cmap)
    # normalise each variable independently
    d1_norm = (dim1_vals - dim1_vals.min()) / (dim1_vals.max() - dim1_vals.min())
    d2_norm = (dim2_vals - dim2_vals.min()) / (dim2_vals.max() - dim2_vals.min())

    rgba = cmap(d1_norm)
    a_min, a_max = alpha_range
    rgba[:, 3] = a_min + d2_norm * (a_max - a_min)  # Set alpha by dim 2

    # use lightness instead of alpha
    # rgb = colors.rgb_to_hsv(rgba[:, :3])
    # rgb[:, 2] = a_min + d2_norm * (a_max - a_min)  # value channel
    # rgba[:, :3] = colors.hsv_to_rgb(rgb)

    return rgba.astype(np.float32)
