"""Class implementing the Supervised Quadratic Feature Analysis (SQFA) model."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations

from ._optim import fitting_loop
from .constraints import FixedFilters, Identity, Sphere
from .distances import _matrix_subset_distance_generator, affine_invariant_sq
from .linalg_utils import conjugate_matrix

__all__ = ["SQFA"]


def __dir__():
    return __all__


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
        self.register_buffer(
            "input_covariances", torch.as_tensor(input_covariances, dtype=torch.float32)
        )
        feature_noise_mat = torch.as_tensor(
            feature_noise, dtype=torch.float32
        ) * torch.eye(n_filters)
        self.register_buffer("diagonal_noise", feature_noise_mat)
        self.constraint = constraint
        self._add_constraint(constraint=self.constraint)

    def get_feature_covariances(self):
        """
        Transform input covariances to filter response covariances.

        Returns
        -------
        torch.Tensor shape (n_classes, n_filters, n_filters)
            Covariances of the transformed features.
        """
        transformed_covariances = conjugate_matrix(self.input_covariances, self.filters)
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

    def fit(
        self,
        distance_fun=None,
        epochs=10,
        lr=0.1,
        pairwise=False,
        **kwargs,
    ):
        """
        Fit the model.

        Parameters
        ----------
        distance_fun : callable
            Function to compute the distance between the transformed feature
            covariances. Should take as input two tensors of shape
            (n_classes, n_filters, n_filters) and return a matrix
            of shape (n_classes, n_classes) with the pairwise distances
            (or squared distances or similarities).
            If None, then the Affine Invariant squared distance is used.
        epochs : int
            Number of epochs to train the model. Default is 100.
        lr : float
            Learning rate for the optimizer. Default is 0.1.
        decay_step : int
            Step at which to decay the learning rate. Default is 1000.
        decay_rate : float
            Rate at which to decay the learning rate. Default is 1.0.
        pairwise : bool
            If True, then filters are optimized pairwise (the first 2 filters
            are optimized together, then held fixed and the next 2 filters are
            optimized together, etc.). If False, all filters are optimized
            together. Default is False.
        **kwargs
            Additional keyword arguments passed to the NAdam optimizer.
        """
        if distance_fun is None:
            distance_fun = affine_invariant_sq

        if not pairwise:
            loss, training_time = fitting_loop(
              model=self,
              distance_fun=distance_fun,
              epochs=epochs,
              lr=lr,
              **kwargs
            )
            return loss, training_time

        else:
            n_pairs = self.filters.shape[0] // 2

            # Require n_pairs to be even
            if self.filters.shape[0] % 2 != 0:
                raise ValueError(
                    "Number of filters must be even for pairwise training."
                )

            # Loop over pairs
            loss = torch.tensor([])
            for i in range(n_pairs):
                # Make function to only evaluate distance on subset of filters
                max_ind = min([2 * i + 2, self.filters.shape[0]])
                inds_filters_used = torch.arange(max_ind)
                distance_subset = _matrix_subset_distance_generator(
                    subset_inds=inds_filters_used, distance_fun=distance_fun
                )

                # Fix the filters already trained
                if i > 0:
                    register_parametrization(
                        self, "filters", FixedFilters(n_row_fixed=i * 2)
                    )

                # Train the current pair
                loss_pair, training_time = fitting_loop(
                  model=self,
                  distance_fun=distance_subset,
                  epochs=epochs,
                  lr=lr,
                  **kwargs
                )

                # Remove fixed filter parametrization
                remove_parametrizations(self, "filters")
                self._add_constraint(constraint=self.constraint)
                loss = torch.cat((loss, loss_pair))

            return loss, training_time

    def _add_constraint(self, constraint="none"):
        """
        Add constraint to the filters.

        Parameters
        ----------
        constraint : str
            Constraint to apply to the filters. Can be 'none', 'sphere' or
            'orthogonal'. Default is 'none'.
        """
        if constraint == "none":
            register_parametrization(self, "filters", Identity())
        elif constraint == "sphere":
            register_parametrization(self, "filters", Sphere())
        elif constraint == "orthogonal":
            orthogonal(self, "filters")
