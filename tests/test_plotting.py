"""Test functions for plotting."""

import matplotlib.pyplot as plt
import pytest
import torch

import sqfa
from make_examples import rotated_classes_dataset, make_dataset_points


@pytest.fixture(scope="function")
def make_dataset():
    """Create a dataset of 4 classes with rotated covariances in 8 dimensions."""
    class_covariances = rotated_classes_dataset()
    class_means = torch.zeros(5, 8)
    return class_covariances, class_means


@pytest.mark.parametrize(
    "dim_pairs",
    [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[0, 2], [4, 6]],
    ],
)
def test_ellipse_plotting(make_dataset, dim_pairs):
    """Test the training function in sqfa._optim."""
    class_covariances, class_means = make_dataset
    figsize = (6, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for i in range(2):
        sqfa.plot.statistics_ellipses(
            ellipses=class_covariances,
            centers=class_means,
            dim_pair=dim_pairs[i],
            ax=ax[i],
        )
    plt.close(fig)


def test_data_wrangle(make_dataset):
    """Test the data wrangling functions."""
    class_covariances, class_means = make_dataset

    subset_stats = sqfa.plot._data_wrangle.statistics_dim_subset(
        class_means, class_covariances, [0, 2, 4]
    )

    assert subset_stats[0].shape == (5, 3), \
        "Subsampling statistics dimensions failed."
    assert subset_stats[1].shape == (5, 3, 3), \
        "Subsampling statistics covariances failed."

    n_points = 20
    points, labels = make_dataset_points(
        n_points=n_points, class_covariances=class_covariances
    )

    keep_classes = torch.tensor([0, 2])
    sub_points, sub_labels = sqfa.plot._data_wrangle.subsample_classes(
        points, labels, classes_to_keep=keep_classes,
    )
    assert torch.all(torch.isin(sub_labels, keep_classes)), \
        "Class subsampling failed."
    assert sub_points.shape[0] == n_points * len(keep_classes), \
        "Subsampled points do not match expected number."

    sub_cl_points, sub_cl_lab = sqfa.plot._data_wrangle.subsample_class_points(
        points, labels, n_per_class=6
    )

    n_classes_sub = len(sub_cl_lab.unique())
    assert n_classes_sub == 5, \
        "Subsampled classes do not match expected number."
    assert sub_cl_points.shape[0] == 6 * n_classes_sub, \
        "Subsampled points do not match expected number per class."


def test_points_plotting(make_dataset):
    class_covariances, class_means = make_dataset

    n_points = 10
    points, labels = make_dataset_points(
        n_points=n_points, class_covariances=class_covariances
    )

    ax = sqfa.plot.scatter_data(
        data=points, labels=labels, dim_pair=(0,2)
    )
    plt.close(ax.figure)
