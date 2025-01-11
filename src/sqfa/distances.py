"""Distances between Symmetric Positive Definite matrices."""

import torch

from .linalg import (
    generalized_eigenvalues,
    spd_log,
)

__all__ = [
    "affine_invariant_sq",
    "affine_invariant",
    "log_euclidean_sq",
    "log_euclidean",
    "fisher_rao_lower_bound",
]


def __dir__():
    return __all__


EPSILON = 1e-6  # Value added inside of square roots


def affine_invariant_sq(A, B):
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


def affine_invariant(A, B):
    """
    Compute the affine invariant distance between SPD matrices.
    A small epsilon is added inside the square root to avoid gradient
    instabilities.

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
    return torch.sqrt(affine_invariant_sq(A, B) + EPSILON)


def log_euclidean_sq(A, B):
    """
    Compute the squared log-Euclidean distance between SPD matrices.

    Parameters
    ----------
    A : torch.Tensor
        Shape (n_batch_A, n_dim, n_dim), the first SPD matrix.
    B : torch.Tensor
        Shape (n_batch_B, n_dim, n_dim), the second SPD matrix.

    Returns
    -------
    distance_squared : torch.Tensor
        Shape (n_batch_A, n_batch_B), the squared log-Euclidean distance.
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    # Compute the log of the matrices
    log_A = spd_log(A)
    log_B = spd_log(B)
    # Compute the squared Frobenius norm of the difference
    diff = log_A[:, None, ...] - log_B[None, ...]
    distance_squared = torch.sum(diff**2, axis=(-2, -1))
    return torch.squeeze(distance_squared)


def log_euclidean(A, B):
    """
    Compute the log-Euclidean distance between SPD matrices.
    A small epsilon is added inside the square root to avoid gradient
    instabilities.

    Parameters
    ----------
    A : torch.Tensor
        Shape (n_batch_A, n_dim, n_dim), the first SPD matrix.
    B : torch.Tensor
        Shape (n_batch_B, n_dim, n_dim), the second SPD matrix.

    Returns
    -------
    distance : torch.Tensor
        Shape (n_batch_A, n_batch_B), the log-Euclidean distance.
    """
    return torch.sqrt(log_euclidean_sq(A, B) + EPSILON)


def _matrix_subset_distance_generator(subset_inds, distance_fun):
    """
    Generate a function takes a sub-matrix from input SPD matrices
    and then computes the distance between them.

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


def _embed_gaussian(means, covariances):
    """
    Embed the parameters of the Gaussian distribution in SPD,
    by stacking the means and the covariances in the format
    [covariances, means;
    means.T, 1].

    Parameters
    ----------
    means : torch.Tensor
        Shape (n_classes, n_filters), the means.
    covariances : torch.Tensor
        Shape (n_classes, n_filters, n_filters), the covariance matrices.

    Returns
    -------
    embedding : torch.Tensor
        Shape (n_classes, n_filters+1, n_filters+1), the embedded SPD matrices.
    """
    n_classes, n_filters = means.shape

    mean_outer_prod = torch.einsum("ni,nj->nij", means, means)
    second_moments = covariances + mean_outer_prod

    embedding = torch.cat([second_moments, means.unsqueeze(1)], dim=1)
    one = torch.ones(n_classes, dtype=means.dtype, device=means.device)
    means_long = torch.cat([means, one.unsqueeze(1)], dim=1)
    embedding = torch.cat([embedding, means_long.unsqueeze(2)], dim=2)
    return embedding


def fisher_rao_lower_bound(means, covariances):
    """
    Compute the lower bound of the Fisher-Rao distance between Gaussians.

    Parameters
    ----------
    means : torch.Tensor
        Shape (n_classes, n_filters), the means.
    covariances : torch.Tensor
        Shape (n_classes, n_filters, n_filters), the covariance matrices.

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the lower bound of the Fisher-Rao distance.
    """
    embedding = _embed_gaussian(means, covariances)
    return affine_invariant(embedding, embedding)
