"""Scatter data of different classes with color code."""

import matplotlib.pyplot as plt
import numpy as np

from ._data_wrangle import subsample_class_points, subsample_classes
from .colors import get_class_rgba, get_normalized_color_map


def scatter_data(
    data,
    labels,
    ax=None,
    values=None,
    dim_pair=(0, 1),
    n_points=1000,
    classes_plot=None,
    legend_type="none",
    **kwargs,
):
    """
    Plot scatter of the data to different categories.

    Parameters
    ----------
    data : torch.Tensor
        Responses to the stimuli. Shape (n_stimuli, n_filters).
    labels : torch.int64
        Class labels of each point with shape (n_points).
    ax : matplotlib.axes.Axes, optional
        Axes to plot the scatter. If None, a new figure is created.
        The default is None.
    values : torch.Tensor, optional
        Values to color the classes. The default is linearly spaced values
        between -1 and 1.
    dim_pair : tuple, optional
        Pair of filters to plot. The default is (0, 1).
    n_points : int, optional
        Number of points per class to plot. The default is 1000.
    classes_plot : list, optional
        List of classes to plot. The default is all classes.
    legend_type : str, optional
        Type of legend to add: 'none', 'continuous', 'discrete'.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Subsample points
    data_plt, labels_plt = subsample_class_points(data, labels, n_points)
    data_plt = data_plt[:, dim_pair].detach().cpu().numpy()
    labels_plt = labels_plt.detach().cpu().numpy()
    data_plt, labels_plt = subsample_classes(data_plt, labels_plt, classes_plot)

    values = np.arange(len(np.unique(labels_plt))) if values is None else values.numpy()

    color_map = plt.get_cmap("viridis")
    class_colors = get_class_rgba(color_map, values)

    ax.scatter(
        data_plt[:, 0],
        data_plt[:, 1],
        c=class_colors[labels_plt],
        s=10,
        alpha=0.5,
    )

    ax.set_xlabel(f"Dimension {dim_pair[0] + 1}")
    ax.set_ylabel(f"Dimension {dim_pair[1] + 1}")

    if legend_type == "continuous":
        color_map, norm = get_normalized_color_map(color_map, values)
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, **kwargs)

    elif legend_type == "discrete":
        for _, class_ind in enumerate(np.unique(labels)):
            ax.scatter([], [], c=[class_colors[class_ind]], label=values[class_ind])
        ax.legend(**kwargs)

    return ax
