"""Tests for the distances module."""
import geomstats.geometry.spd_matrices as spd_matrices
import pytest
import torch

from make_examples import sample_spd
from sqfa.distances import (
    affine_invariant_sq,
)


def affine_invariant_ref(A, B):
    """
    Compute the affine invariant distance between pairs (A, B) using geomstats.

    Parameters
    ----------
    A : np.ndarray
        A tensor of spd matrices of shape (n_matrices_A, n_dim, n_dim).
    B : np.ndarray
        A tensor of spd matrices of shape (n_matrices_B, n_dim, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    if B.dim() < 3:
        B = B.unsqueeze(0)

    n_dim = A.shape[1]

    # Initialize manifold
    manifold = spd_matrices.SPDMatrices(n=n_dim, equip=spd_matrices.SPDAffineMetric)

    distances_geomstats = manifold.metric.squared_dist(
      point_a = A.numpy()[:,None,:,:], point_b = B.numpy()[None,:,:,:]
    )

    return torch.squeeze(torch.as_tensor(distances_geomstats, dtype=torch.float32))


@pytest.fixture(scope="function")
def sample_spd_matrices(n_matrices_A, n_matrices_B, n_dim):
    """Generate a tensor of SPD matrices."""
    A = sample_spd(n_matrices_A, n_dim)
    B = sample_spd(n_matrices_B, n_dim)
    return A, B


@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_generalized_eigenvalues(sample_spd_matrices, n_matrices_A,
                                 n_matrices_B, n_dim):
    """Test the generalized eigenvalues function."""
    A, B = sample_spd_matrices

    distances = affine_invariant_sq(A, B)
    distances_geomstats = affine_invariant_ref(A, B)

    assert torch.allclose(distances, distances_geomstats, atol=1e-5)
