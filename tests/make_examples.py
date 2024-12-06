"""Generate synthetic data for testing."""

import torch

def make_orthogonal_matrices(n_matrices, n_dim):
    """Generate random orthogonal matrices."""
    low_tri = torch.randn(n_matrices, n_dim, n_dim)
    low_tri = torch.tril(low_tri, diagonal=-1)
    skew_sym = low_tri - low_tri.transpose(1, 2)
    orthogonal = torch.matrix_exp(skew_sym)
    return orthogonal

def sample_spd(n_matrices, n_dim):
    """Generate random SPD matrices."""
    eigvals = 2 * (torch.rand(n_matrices, n_dim)) ** 2 + 0.01
    eigvecs = make_orthogonal_matrices(n_matrices, n_dim)
    spd = torch.einsum('ijk,ik,ikl->ijl', eigvecs, eigvals, eigvecs.transpose(1, 2))
    return torch.squeeze(spd)
