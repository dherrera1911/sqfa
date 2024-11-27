import torch
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance

__all__ = ["class_statistics"]


def class_statistics(points, labels, estimator="empirical", shrunk_param=0.1):
    """
    Compute the mean, covariance and second moment matrix of each class.

    Parameters
    ----------
    points : torch.Tensor
        Data points with shape (n_points, n_dim).
    labels : torch.Tensor
        Class labels of each point with shape (n_points).
    estimator:
        Covariance estimator to use. Options are "empirical",
        "ledoit-wolf", "oas" and "shrunk". Default is "empirical".

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
        n_points = torch.tensor(class_points.shape[0], dtype=dtype)

        means[i] = torch.mean(class_points, dim=0)

        if estimator == "empirical":
            cov_i = EmpiricalCovariance(store_precision=False).fit(class_points).covariance_
        elif estimator == "ledoit-wolf":
            cov_i = LedoitWolf(store_precision=False).fit(class_points).covariance_
        elif estimator == "oas":
            cov_i = OAS(store_precision=False).fit(class_points).covariance_
        elif estimator == "shrunk":
            cov_i = ShrunkCovariance(shrinkage=shrunk_param).fit(class_points).covariance_
        covariances[i] = torch.tensor(cov_i, dtype=dtype)
        second_moments[i] = covariances[i] + torch.einsum('i,j->ij', means[i], means[i])

    statistics_dict = {
        "means": means,
        "covariances": covariances,
        "second_moments": second_moments,
    }
    return statistics_dict
