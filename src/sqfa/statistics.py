"""Functions to compute class statistics of labeled data points."""

import torch

__all__ = ["class_statistics", "oas_covariance"]


def class_statistics(points, labels, estimator="empirical"):
    """
    Compute the mean, covariance and second moment matrix of each class.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).
    labels : torch.Tensor
        Class labels of each point with shape (n_points).
    estimator:
        Covariance estimator to use. Options are "empirical"
        and "oas". Default is "empirical".

    Returns
    -------
    statistics_dict : dict
        Dictionary containing the mean, covariance and second moment matrix
        of each class.
    """
    dtype = points.dtype
    n_classes = int(torch.max(labels) + 1)
    n_dim = points.shape[-1]

    means = torch.zeros(n_classes, n_dim, dtype=dtype)
    covariances = torch.zeros(n_classes, n_dim, n_dim, dtype=dtype)
    second_moments = torch.zeros(n_classes, n_dim, n_dim, dtype=dtype)

    for i in range(n_classes):
        indices = (labels == i).nonzero().squeeze(1)
        class_points = points[indices]

        means[i] = torch.mean(class_points, dim=0)

        if estimator == "empirical":
            cov_i = sample_covariance(class_points)
        elif estimator == "oas":
            cov_i = oas_covariance(class_points)
        covariances[i] = torch.as_tensor(cov_i, dtype=dtype)
        second_moments[i] = covariances[i] + torch.einsum("i,j->ij", means[i], means[i])

    statistics_dict = {
        "means": means,
        "covariances": covariances,
        "second_moments": second_moments,
    }
    return statistics_dict


def oas_covariance(points, assume_centered=False):
    """
    Compute the OAS covariance matrix of the given points.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).

    Returns
    -------
    cov : torch.Tensor
        OAS covariance matrix of the points.

    References
    ----------
    .. [1] :arxiv:`"Shrinkage algorithms for MMSE covariance estimation.",
           Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
           IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
           <0907.4698>`
    """
    n_samples = points.shape[0]
    n_dim = points.shape[1]

    sample_cov = sample_covariance(points, assume_centered=assume_centered)

    # Compute the OAS shrinkage parameter
    tr_cov = torch.trace(sample_cov)
    tr_prod = torch.sum(sample_cov**2)
    shrinkage = ((1 - 2 / n_dim) * tr_prod + tr_cov**2) / (
        (n_samples + 1 - 2 / n_dim) * (tr_prod - tr_cov**2 / n_dim)
    )
    shrinkage = min(1.0, shrinkage)

    # Compute the OAS covariance matrix
    target = torch.eye(n_dim) * torch.trace(sample_cov) / n_dim
    oas_cov = (1 - shrinkage) * sample_cov + shrinkage * target
    return oas_cov


def sample_covariance(points, assume_centered=False):
    """
    Compute the sample covariance matrix of the given points.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).
    assume_centered : bool
        If True, assume that the data is centered. Default is False.

    Returns
    -------
    cov : torch.Tensor
        Sample covariance matrix of the points.
    """
    n_points, n_dim = points.shape

    if assume_centered:
        cov = torch.einsum("ij,ik->jk", points, points) / n_points
    else:
        mean = torch.mean(points, dim=0)
        centered_points = points - mean
        cov = torch.einsum("ij,ik->jk", centered_points, centered_points) / (
            n_points - 1
        )

    return cov
