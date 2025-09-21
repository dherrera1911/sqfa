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
    "fisher_rao_lower_bound_sq",
    "bhattacharyya",
    "mahalanobis_sq",
    "mahalanobis",
    "hellinger",
]


def __dir__():
    return __all__


EPSILON = 1e-6  # Value added inside of square roots


def _unsqueeze_mean(A):
    """Add batch dimension to the mean if it is not present."""
    if A.dim() == 1:
        A = A.unsqueeze(0)
    return A


def _unsqueeze_covariance(A):
    """Add batch dimension to the covariance if it is not present."""
    if A.dim() == 2:
        A = A.unsqueeze(0)
    return A


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
    A = _unsqueeze_covariance(A)
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


def _embed_gaussian(statistics):
    """
    Embed the parameters of the Gaussian distribution in SPD,
    by stacking the means and the covariances in the format
    [covariances, means;
    means.T, 1].

    Parameters
    ----------
    statistics: dict
        Dictionary containing the means and covariances of the Gaussian
        distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    embedding : torch.Tensor
        Shape (n_classes, n_filters+1, n_filters+1), the embedded SPD matrices.
    """
    means = _unsqueeze_mean(statistics["means"])
    covariances = _unsqueeze_covariance(statistics["covariances"])

    n_classes, n_filters = means.shape

    mean_outer_prod = torch.einsum("ni,nj->nij", means, means)
    second_moments = covariances + mean_outer_prod

    embedding = torch.cat([second_moments, means.unsqueeze(1)], dim=1)
    one = torch.ones(n_classes, dtype=means.dtype, device=means.device)
    means_long = torch.cat([means, one.unsqueeze(1)], dim=1)
    embedding = torch.cat([embedding, means_long.unsqueeze(2)], dim=2)
    return embedding


def fisher_rao_lower_bound_sq(statistics_A, statistics_B):
    """
    Compute the Calvo & Oller lower bound of the Fisher-Rao squared
    distance between Gaussians.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance_squared : torch.Tensor
        Shape (n_classes, n_classes), the lower bound of the Fisher-Rao squared
        distance.
    """
    embedding_A = _embed_gaussian(statistics_A)
    embedding_B = _embed_gaussian(statistics_B)
    distance_squared = affine_invariant_sq(embedding_A, embedding_B) / 2
    return distance_squared


def fisher_rao_lower_bound(statistics_A, statistics_B):
    """
    Compute the Calvo & Oller lower bound of the Fisher-Rao squared
    distance between Gaussians.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the lower bound of the Fisher-Rao distance.
    """
    distances_squared = fisher_rao_lower_bound_sq(statistics_A, statistics_B)
    return torch.sqrt(distances_squared + EPSILON)


def bhattacharyya(statistics_A, statistics_B):
    """
    Compute the Bhattacharyya distance between two Gaussian distributions.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the lower bound of the Fisher-Rao distance.
    """
    mean_A = _unsqueeze_mean(statistics_A["means"])
    cov_A = _unsqueeze_covariance(statistics_A["covariances"])
    mean_B = _unsqueeze_mean(statistics_B["means"])
    cov_B = _unsqueeze_covariance(statistics_B["covariances"])

    mean_cov_inv = torch.linalg.inv((cov_A[:, None] + cov_B[None, :]) / 2)
    means_diff = mean_A[:, None] - mean_B[None, :]
    term1 = torch.einsum("ijk,ijkl,ijl->ij", means_diff, mean_cov_inv, means_diff)

    mean_cov = (cov_A[:, None] + cov_B[None, :]) * 0.5
    cov_det_A = torch.logdet(cov_A)
    cov_det_B = torch.logdet(cov_B)
    term2 = torch.logdet(mean_cov) - (cov_det_A[:, None] + cov_det_B[None, :]) * 0.5

    dist = term1 * 1 / 8 + term2 * 0.5
    return torch.squeeze(dist)


def mahalanobis_sq(statistics_A, statistics_B):
    """
    Compute the squared Mahalanobis distance between Gaussian distributions,
    using the mean covariance matrix for each pair.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance_squared : torch.Tensor
        Shape (n_classes, n_classes), the squared Mahalanobis distance.
    """
    mean_A = _unsqueeze_mean(statistics_A["means"])  # (n_classes, n_filters)
    cov_A = _unsqueeze_covariance(
        statistics_A["covariances"]
    )  # (n_classes, n_filters, n_filters)

    mean_B = _unsqueeze_mean(statistics_B["means"])  # (n_classes, n_filters)
    cov_B = _unsqueeze_covariance(
        statistics_B["covariances"]
    )  # (n_classes, n_filters, n_filters)

    mean_cov = (
        cov_A[:, None] + cov_B[None, :]
    ) / 2  # (n_classes, n_classes, n_filters, n_filters)
    mean_cov_inv = torch.linalg.inv(mean_cov)  # Invert mean covariance

    means_diff = mean_A[:, None] - mean_B[None, :]  # (n_classes, n_classes, n_filters)

    # Compute squared Mahalanobis distance
    distance_squared = torch.einsum(
        "ijk,ijkl,ijl->ij", means_diff, mean_cov_inv, means_diff
    )

    return distance_squared


def mahalanobis(statistics_A, statistics_B):
    """
    Compute the Mahalanobis distance between Gaussian distributions,
    using the mean covariance matrix for each pair.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the Mahalanobis distance.
    """
    distances_squared = mahalanobis_sq(statistics_A, statistics_B)
    return torch.sqrt(distances_squared + EPSILON)


def hellinger(statistics_A, statistics_BB):
    """
    Compute the Hellinger distance between two Gaussian distributions.
    An epsilon is added inside the square root to avoid gradient
    instabilities.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the lower bound of the Fisher-Rao distance.
    """
    dist = bhattacharyya(statistics_A, statistics_B)
    hellinger = torch.sqrt(1 - torch.exp(-dist) + EPSILON)
    return hellinger


def fisher_rao_same_cov(statistics_A, statistics_B):
    """
    Compute the exact Fisher-Rao distance between two Gaussian
    distributions with the same covariance matrix. The covariance matrix is
    taken as the mean of the two covariance matrices.

    Parameters
    ----------
    statistics_A: dict
        Dictionary containing the means and covariances of the first
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    statistics_B: dict
        Dictionary containing the means and covariances of the second
        Gaussian distribution, with keys "means" and "covariances".
        - The means are a torch.Tensor of shape (n_classes, n_filters)
        - The covariances are a torch.Tensor of
          shape (n_classes, n_filters, n_filters).

    Returns
    -------
    distance : torch.Tensor
        Shape (n_classes, n_classes), the Fisher-Rao distance.

    References
    ----------
    .. [1] F. Nielsen, "A Simple Approximation Method for the Fisher–Rao
    Distance between Multivariate Normal Distributions"
    Entropy, vol. 25, no. 4, pp. 654, 2023.
    """
    dist_maha_sq = mahalanobis_sq(statistics_A, statistics_B)
    fr_dist = torch.sqrt(torch.tensor(2.0)) * \
        torch.acosh(1 + dist_maha_sq / 4)
    return fr_dist
