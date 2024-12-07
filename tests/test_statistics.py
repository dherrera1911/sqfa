"""Tests for the statistics module."""

import torch

from sqfa.statistics import class_statistics


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
