"""Test functions for plotting."""

import matplotlib.pyplot as plt
import pytest
import torch

import sqfa
from make_examples import rotated_classes_dataset

MAX_EPOCHS = 100


@pytest.fixture(scope="function")
def make_dataset():
    """Create a dataset of 4 classes with rotated covariances in 8 dimensions."""
    class_covariances = rotated_classes_dataset()
    class_means = torch.zeros(4, 8)
    return class_covariances, class_means


@pytest.mark.parametrize("dim_pairs", [
[[0, 1], [2, 3]],
[[4, 5], [6, 7]],
[[0, 2], [4, 6]],
])
def test_ellipse_plotting(make_dataset, dim_pairs):
    """Test the training function in sqfa._optim."""
    class_covariances, class_means = make_dataset
    figsize = (6, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for i in range(2):
        sqfa.plot.statistics_ellipses(ellipses=class_covariances,
                                      centers=class_means,
                                      dim_pair=dim_pairs[i], ax=ax[i])
    plt.close(fig)
