"""Distances between Symmetric Positive Definite matrices."""

import torch

from .linalg_utils import (
    generalized_eigenvalues,
)

__all__ = ["ai_distance_sq"]


def __dir__():
    return __all__


def ai_distance_sq(A, B):
    """
    Compute the squared affine invariant distance between SPD matrices.

    Parameters
    ----------
    A : torch.Tensor
        Shape (n_batch_A, n_dim, n_dim), the first SPD matrix.
    B : torch.Tensor
        Shape (n_batch_B, n_dim, n_dim), the second SPD matrix.

    Returns
    -------
    distance_squared : torch.Tensor
        Shape (n_batch_A, n_batch_B), the squared affine invariant distance.
    """
    # Compute the generalized eigenvalues
    gen_eigvals = generalized_eigenvalues(A, B)
    # Compute the distance
    distance_squared = torch.sum(torch.log(gen_eigvals) ** 2, axis=-1)
    return distance_squared


def ai_distance(A, B):
    """
    Compute the affine invariant distance between SPD matrices.

    Parameters
    ----------
    A : torch.Tensor
        Shape (n_batch_A, n_dim, n_dim), the first SPD matrix.
    B : torch.Tensor
        Shape (n_batch_B, n_dim, n_dim), the second SPD matrix.

    Returns
    -------
    distance : torch.Tensor
        Shape (n_batch_A, n_batch_B), the affine invariant distance.
    """
    return torch.sqrt(ai_distance_sq(A, B))
