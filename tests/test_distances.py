"""Tests for the distances module."""

import pytest
import torch

from make_examples import sample_spd
from sqfa.distances import (
    affine_invariant_sq,
    fisher_rao_lower_bound_sq,
    log_euclidean_sq,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="function")
def sample_spd_matrices(n_classes, n_dim):
    """Generate a tensor of SPD matrices."""
    spd_mat = sample_spd(n_classes, n_dim)
    return spd_mat


@pytest.fixture(scope="function")
def sample_vectors(n_classes, n_dim):
    """Generate a tensor of vectors."""
    A = torch.randn(n_classes, n_dim)
    return A


def get_diag(A):
    """Get the diagonal of a tensor, even when tensor is a scalar."""
    if A.dim() > 0:
        return A.diagonal(dim1=-2, dim2=-1)
    else:
        return A


@pytest.mark.parametrize("n_classes", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_distance_sq(sample_spd_matrices, n_classes, n_dim):
    """Test the generalized eigenvalues function."""
    spd_mat = sample_spd_matrices

    ai_distances_sq = affine_invariant_sq(spd_mat, spd_mat)

    if n_classes != 1:
        assert ai_distances_sq.shape == (n_classes, n_classes)
    else:
        assert ai_distances_sq.shape == ()

    assert torch.allclose(
        ai_distances_sq, ai_distances_sq.T, atol=1e-5
    ), "The self-distance matrix for AIRM is not symmetric"

    assert torch.allclose(
        get_diag(ai_distances_sq), torch.zeros(n_classes), atol=1e-5
    ), "The diagonal of the self-distance matrix for AIRM is not zero"

    spd_inv = torch.inverse(spd_mat)

    ai_distances_inv_sq = affine_invariant_sq(spd_inv, spd_inv)

    assert torch.allclose(
        ai_distances_sq, ai_distances_inv_sq, atol=1e-5
    ), "The affine invariant distance is not invariant to inversion."

    le_distances_sq = log_euclidean_sq(spd_mat, spd_mat)

    if n_classes != 1:
        assert le_distances_sq.shape == (n_classes, n_classes)
    else:
        assert le_distances_sq.shape == ()

    assert torch.allclose(
        le_distances_sq, le_distances_sq.T, atol=1e-5
    ), "The self-distance matrix for AIRM is not symmetric"

    assert torch.allclose(
        get_diag(le_distances_sq), torch.zeros(n_classes), atol=1e-5
    ), "The diagonal of the self-distance matrix for AIRM is not zero"

    le_distances_inv_sq = log_euclidean_sq(spd_inv, spd_inv)

    assert torch.allclose(
        le_distances_sq, le_distances_inv_sq, atol=1e-5
    ), "The log-Euclidean distance is not invariant to inversion."

    eye = torch.eye(n_dim)

    ai_dist_to_eye = affine_invariant_sq(spd_mat, eye)
    le_dist_to_eye = log_euclidean_sq(spd_mat, eye)

    assert torch.allclose(
        ai_dist_to_eye, le_dist_to_eye, atol=1e-5
    ), "The AIRM and LE distances from the identity are not equal."


@pytest.mark.parametrize("n_classes", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_fisher_rao_sq(sample_spd_matrices, sample_vectors, n_classes, n_dim):
    """Test the generalized eigenvalues function."""
    spd_mat = sample_spd_matrices
    means = sample_vectors
    stats_dict = {
        "means": means,
        "covariances": spd_mat,
    }

    fr_distances = fisher_rao_lower_bound_sq(stats_dict, stats_dict)

    if n_classes != 1:
        assert fr_distances.shape == (n_classes, n_classes)
    else:
        assert fr_distances.shape == ()

    assert torch.allclose(
        fr_distances, fr_distances.T, atol=1e-5
    ), "The self-distance matrix for AIRM is not symmetric"

    assert torch.allclose(
        get_diag(fr_distances), torch.zeros(n_classes), atol=1e-5
    ), "The diagonal of the self-distance matrix for AIRM is not zero"
