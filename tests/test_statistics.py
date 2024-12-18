"""Tests for the statistics module."""

import pytest
import torch

from make_examples import rotated_classes_dataset
from sqfa.statistics import class_statistics, pca, pca_from_scatter


@pytest.fixture(scope="function")
def make_pca_points(n_points, n_dim):
    """Create a dataset of n_points with n_dims to test PCA."""
    points = torch.randn(n_points, n_dim)
    return points


@pytest.fixture(scope="function")
def make_dataset():
    """Create a dataset of 8 classes with rotated covariances."""
    class_covariances = rotated_classes_dataset()
    return class_covariances


def test_class_statistics():
    """Test function that computes class-specific statistics."""
    n_points = 1000
    X = torch.ones(n_points, 4)
    y = torch.randint(0, 3, (n_points,))
    class_stats = class_statistics(X, y, estimator="empirical")
    assert class_stats["means"].shape == (3, 4), "Means have incorrect shape."
    assert class_stats["covariances"].shape == (
        3,
        4,
        4,
    ), "Covariances have incorrect shape."
    assert class_stats["second_moments"].shape == (
        3,
        4,
        4,
    ), "Covariances have incorrect shape."

    assert torch.allclose(
        class_stats["means"], torch.ones(3, 4), atol=1e-6
    ), "Means are not correct."
    assert torch.allclose(
        class_stats["covariances"], torch.zeros(3, 4, 4), atol=1e-5
    ), "Covariances are not correct."
    assert torch.allclose(
        class_stats["second_moments"], torch.ones(3, 4, 4), atol=1e-5
    ), "Counts are not correct."


@pytest.mark.parametrize("n_points", [1000])
@pytest.mark.parametrize("n_dim", [2, 10])
@pytest.mark.parametrize("n_components", [None, 2, 4])
def test_pca_points(make_pca_points, n_points, n_dim, n_components):
    """Test PCA on a dataset of n_points with n_dims."""
    points = make_pca_points

    # Check that error is raised if n_components > n_dim
    if n_components is not None and n_components > n_dim:
        with pytest.raises(ValueError):
            pca(points, n_components)
        return
    else:
        components = pca(points, n_components)

    if n_components is None:
        n_components = min(n_dim, n_points)

    pca_projections = points @ components.T
    variances = torch.var(pca_projections, dim=0)

    assert variances[0] >= variances[-1], "Variance is not decreasing."
    assert components.shape[0] == n_components, "Components have incorrect shape."


@pytest.mark.parametrize("n_components", [None, 2, 4, 10])
def test_pca_scatters(make_dataset, n_components):
    """Test PCA on a dataset of n_points with n_dims."""
    class_covariances = make_dataset

    if n_components is not None and n_components > class_covariances.shape[-1]:
        with pytest.raises(ValueError):
            pca_from_scatter(class_covariances, n_components)
        return
    else:
        components = pca_from_scatter(class_covariances, n_components)

    if n_components is None:
        n_components = class_covariances.shape[-1]

    mean_covariance = torch.mean(class_covariances, dim=0)
    variances = torch.einsum("ij,jk,ki->i", components, mean_covariance, components.T)

    assert variances[0] >= variances[-1], "Variance is not decreasing."
    assert components.shape[0] == n_components, "Components have incorrect shape."
