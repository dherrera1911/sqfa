"""Tests for the distances module."""

import pytest
import torch

from make_examples import sample_spd
from sqfa.distances import (
    affine_invariant_sq,
    log_euclidean_sq,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="function")
def sample_spd_matrices(n_matrices_A, n_dim):
    """Generate a tensor of SPD matrices."""
    A = sample_spd(n_matrices_A, n_dim)
    return A


def get_diag(A):
    """Get the diagonal of a tensor."""
    if A.dim() > 0:
        return A.diagonal(dim1=-2, dim2=-1)
    else:
        return A


@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_distance_sq(sample_spd_matrices, n_matrices_A, n_dim):
    """Test the generalized eigenvalues function."""
    A = sample_spd_matrices

    ai_distances_sq = affine_invariant_sq(A, A)

    if n_matrices_A != 1:
        assert ai_distances_sq.shape == (n_matrices_A, n_matrices_A)
    else:
        assert ai_distances_sq.shape == ()

    assert torch.allclose(
        ai_distances_sq, ai_distances_sq.T, atol=1e-5
    ), "The self-distance matrix for AIRM is not symmetric"

    assert torch.allclose(
        get_diag(ai_distances_sq), torch.zeros(n_matrices_A), atol=1e-5
    ), "The diagonal of the self-distance matrix for AIRM is not zero"

    A_inv = torch.inverse(A)

    ai_distances_inv_sq = affine_invariant_sq(A_inv, A_inv)

    assert torch.allclose(
        ai_distances_sq, ai_distances_inv_sq, atol=1e-5
    ), "The affine invariant distance is not invariant to inversion."

    le_distances_sq = log_euclidean_sq(A, A)

    if n_matrices_A != 1:
        assert le_distances_sq.shape == (n_matrices_A, n_matrices_A)
    else:
        assert le_distances_sq.shape == ()

    assert torch.allclose(
        le_distances_sq, le_distances_sq.T, atol=1e-5
    ), "The self-distance matrix for AIRM is not symmetric"

    assert torch.allclose(
        get_diag(le_distances_sq), torch.zeros(n_matrices_A), atol=1e-5
    ), "The diagonal of the self-distance matrix for AIRM is not zero"

    le_distances_inv_sq = log_euclidean_sq(A_inv, A_inv)

    assert torch.allclose(
        le_distances_sq, le_distances_inv_sq, atol=1e-5
    ), "The log-Euclidean distance is not invariant to inversion."

    B = torch.eye(n_dim)

    ai_dist_to_eye = affine_invariant_sq(A, B)
    le_dist_to_eye = log_euclidean_sq(A, B)

    assert torch.allclose(
        ai_dist_to_eye, le_dist_to_eye, atol=1e-5
    ), "The AIRM and LE distances from the identity are not equal."
