"""Class implementing the Supervised Quadratic Feature Analysis (SQFA) model."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils.parametrize import register_parametrization

from .constrains import Identity, Sphere


class SQFA(nn.Module):
    """Supervised Quadratic Feature Analysis (SQFA) model."""

    def __init__(
        self,
        input_covariances,
        feature_noise=0,
        n_filters=2,
        filters=None,
        constraint="sphere",
    ):
        """
        Initialize SQFA.

        Parameters
        ----------
        input_covariances : torch.Tensor
            Covariance matrices of the input data for each class, of
            shape (n_classes, n_dim, n_dim).
        feature_noise : float
            Noise added to the features outputs, i.e. a diagonal term added
            to the covariance matrix of the features. Default is 0.
        n_filters : int
            Number of filters to use. Default is 2. If filters is provided,
            n_filters is ignored.
        filters : torch.Tensor
            Filters to use. If n_filters is provided, filters are randomly
            initialized. Default is None. Of shape (n_filters, n_dim).
        constraint : str
            Constraint to apply to the filters. Can be 'none', 'sphere' or
            'orthogonal'. Default is 'sphere'.
        """
        super().__init__()
        n_dim = input_covariances.shape[-1]

        if filters is None:
            filters = torch.randn(n_filters, n_dim)
        else:
            filters = torch.as_tensor(filters, dtype=torch.float32)

        self.filters = nn.Parameter(filters)
        self.register_buffer("input_covariances", torch.as_tensor(input_covariances, dtype=torch.float32))
        self.register_buffer("diagonal_noise", feature_noise * torch.eye(filters.shape[0]))
        if constraint == "none":
            register_parametrization(self, "filters", Identity())
        elif constraint == "sphere":
            register_parametrization(self, "filters", Sphere())
        elif constraint == "orthogonal":
            orthogonal(self, "filters")

    def get_feature_covariances(self):
        """
        Transform input covariances to filter response covariances.

        Returns
        -------
        torch.Tensor shape (n_classes, n_filters, n_filters)
            Covariances of the transformed features.
        """
        transformed_covariances = torch.einsum(
            "ij,cjk,km->cim", self.filters, self.input_covariances, self.filters.T
        )
        return transformed_covariances + self.diagonal_noise[None, :, :]

    def forward(self, data_points):
        """
        Transform input data to features.

        Parameters
        ----------
        data_points : torch.Tensor
            Input data of shape (n_samples, n_dim).

        Returns
        -------
        torch.Tensor shape (n_samples, n_filters)
            Data transformed to features.
        """
        transformed_points = torch.einsum("ij,nj->ni", self.filters, data_points)
        return transformed_points
