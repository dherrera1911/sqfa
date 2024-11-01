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


def _matrix_subset_distance_generator(subset_inds, distance_fun):
    """
    Generate a function that computes the distance using only a subset
    of the matrix elements. The distance computed is the same as the one
    computed by `distance_fun`, but only for the subset of elements
    specified by `subset_inds`.

    Parameters
    ----------
    subset_inds : torch.Tensor
        Shape (n_subset,), the indices of the subset of elements.
    distance_fun : callable
        Function to compute the distance between the transformed feature
        covariances. Should take as input two tensors of shape
        (n_classes, n_filters, n_filters) and return a matrix
        of shape (n_classes, n_classes) with the pairwise distances
        (or squared distances or similarities).
        If None, then the Affine Invariant squared distance is used.

    Returns
    -------
    distance_subset : callable
        Function that computes the distance between the subset of elements.
    """
    subset_inds_copy = subset_inds.clone()
    def distance_subset(A, B):
        # Extract the subset of elements
        A_subset = A[:, subset_inds_copy][:, :, subset_inds_copy]
        B_subset = B[:, subset_inds_copy][:, :, subset_inds_copy]
        return distance_fun(A_subset, B_subset)
    return distance_subset
