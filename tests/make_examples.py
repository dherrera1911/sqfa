"""Generate synthetic data for testing."""

import geomstats.geometry.spd_matrices as spd
import torch


def sample_spd(n_matrices, n_dim):
    """Generate random SPD matrices."""
    manifold = spd.SPDMatrices(n=n_dim, equip=spd.SPDAffineMetric)
    covariances = manifold.random_point(n_samples=n_matrices)
    return torch.as_tensor(covariances, dtype=torch.float32)
