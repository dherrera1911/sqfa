"""Class implementing the Supervised Quadratic Feature Analysis (SQFA) model."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations

from ._optim import fitting_loop
from .constraints import FixedFilters, Identity, Sphere
from .distances import _matrix_subset_distance_generator, affine_invariant_sq
from .linalg import conjugate_matrix
from .statistics import class_statistics

__all__ = ["SQFA"]


def __dir__():
    return __all__


class SQFA(nn.Module):
    """Supervised Quadratic Feature Analysis (SQFA) model."""

    def __init__(
        self,
        n_dim,
        feature_noise=0,
        n_filters=2,
        filters=None,
        distance_fun=None,
        constraint="sphere",
    ):
        """
        Initialize SQFA.

        Parameters
        ----------
        n_dim : int
            Dimension of the input data space.
        feature_noise : float
            Noise added to the features outputs, i.e. a diagonal term added
            to the covariance matrix of the features. Default is 0.
        n_filters : int
            Number of filters to use. Default is 2. If filters is provided,
            n_filters is ignored.
        filters : torch.Tensor
            Filters to use. If n_filters is provided, filters are randomly
            initialized. Default is None. Of shape (n_filters, n_dim).
        distance_fun : callable
            Function to compute the distance between the transformed feature
            scatter matrices. Should take as input two tensors of shape
            (n_classes, n_filters, n_filters) and return a matrix
            of shape (n_classes, n_classes) with the pairwise distances
            (or squared distances or similarities).
            If None, then the Affine Invariant squared distance is used.
        constraint : str
            Constraint to apply to the filters. Can be 'none', 'sphere' or
            'orthogonal'. Default is 'sphere'.
        """
        super().__init__()

        if filters is None:
            filters = torch.randn(n_filters, n_dim)
        else:
            filters = torch.as_tensor(filters, dtype=torch.float32)

        self.filters = nn.Parameter(filters)

        feature_noise_mat = torch.as_tensor(
            feature_noise, dtype=torch.float32
        ) * torch.eye(n_filters)
        self.register_buffer("diagonal_noise", feature_noise_mat)

        if distance_fun is None:
            self.distance_fun = affine_invariant_sq
        else:
            self.distance_fun = distance_fun

        self.constraint = constraint
        self._add_constraint(constraint=self.constraint)

    def transform_scatters(self, data_scatters):
        """
        Transform data scatter matrices to feature space scatter matrices.

        Parameters
        ----------
        data_scatters : torch.Tensor
            Tensor of shape (n_classes, n_dim, n_dim).

        Returns
        -------
        torch.Tensor shape (n_classes, n_filters, n_filters)
            Covariances of the transformed features.
        """
        feature_scatters = conjugate_matrix(data_scatters, self.filters)
        return feature_scatters

    def get_class_distances(self, data_scatters, regularized=False):
        """
        Compute the pairwise distances between the feature scatter matrices of the
        different classes.

        Parameters
        ----------
        data_scatters : torch.Tensor
            Tensor of shape (n_classes, n_dim, n_dim).
        regularized : bool
            If True, regularize the distances by adding a small value to the
            diagonal of the transformed scatter matrices. Default is False.

        Returns
        -------
        torch.Tensor shape (n_classes, n_classes)
            Pairwise distances between the transformed feature scatter matrices.
        """
        feature_scatters = self.transform_scatters(data_scatters)

        if regularized:
            feature_scatters = feature_scatters + self.diagonal_noise[None, :, :]

        distances = self.distance_fun(feature_scatters, feature_scatters)
        return distances

    def transform(self, data_points):
        """
        Transform data to feature space.

        Parameters
        ----------
        data_points : torch.Tensor
            Input data of shape (n_samples, n_dim).

        Returns
        -------
        torch.Tensor shape (n_samples, n_filters)
            Data transformed to feature space.
        """
        transformed_points = torch.einsum("ij,nj->ni", self.filters, data_points)
        return transformed_points

    def fit(
        self,
        X=None,
        y=None,
        data_scatters=None,
        max_epochs=300,
        lr=0.1,
        estimator="oas",
        pairwise=False,
        show_progress=True,
        return_loss=False,
        **kwargs,
    ):
        """
        Fit the SQFA model to data using the LBFGS optimizer.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_dim). If data_scatters is None,
            then X and y must be provided.
        y : torch.Tensor
            Labels of shape (n_samples,). If data_scatters is None, then X
            and y must be provided. Labels must be integers starting from 0.
        data_scatters : torch.Tensor
            Tensor of shape (n_classes, n_dim, n_dim) with the second moments
            of the data for each class. If None, then X and y must be provided.
            Default is None.
        max_epochs : int, optional
            Number of max training epochs. By default 50.
        lr : float
            Learning rate for the optimizer. Default is 0.1.
        estimator:
            Covariance estimator to use. Options are "empirical",
            and "oas". Default is "oas".
        pairwise : bool
            If True, then filters are optimized pairwise (the first 2 filters
            are optimized together, then held fixed and the next 2 filters are
            optimized together, etc.). If False, all filters are optimized
            together. Default is False.
        show_progress : bool
            If True, show a progress bar during training. Default is True.
        return_loss : bool
            If True, return the loss after training. Default is False.
        **kwargs
            Additional keyword arguments passed to the NAdam optimizer.
        """
        if data_scatters is None:
            if X is None or y is None:
                raise ValueError("Either data_scatters or X and y must be provided.")
            stats_dict = class_statistics(X, y, estimator=estimator)
            data_scatters = stats_dict["second_moments"]

        if not pairwise:
            loss, training_time = fitting_loop(
                model=self,
                data_scatters=data_scatters,
                max_epochs=max_epochs,
                lr=lr,
                show_progress=show_progress,
                return_loss=True,
                **kwargs,
            )

        else:
            distance_fun_original = self.distance_fun
            n_pairs = self.filters.shape[0] // 2

            # Require n_pairs to be even
            if self.filters.shape[0] % 2 != 0:
                raise ValueError(
                    "Number of filters must be even for pairwise training."
                )

            # Loop over pairs
            loss = torch.tensor([])
            training_time = torch.tensor([])
            for i in range(n_pairs):
                # Make function to only evaluate distance on subset of filters
                max_ind = min([2 * i + 2, self.filters.shape[0]])
                inds_filters_used = torch.arange(max_ind)
                self.distance_fun = _matrix_subset_distance_generator(
                    subset_inds=inds_filters_used, distance_fun=distance_fun_original
                )

                # Fix the filters already trained
                if i > 0:
                    register_parametrization(
                        self, "filters", FixedFilters(n_row_fixed=i * 2)
                    )

                # Train the current pair
                loss_pair, training_time = fitting_loop(
                    model=self,
                    data_scatters=data_scatters,
                    max_epochs=max_epochs,
                    lr=lr,
                    show_progress=show_progress,
                    return_loss=True,
                    **kwargs,
                )

                # Remove fixed filter parametrization
                remove_parametrizations(self, "filters")
                self._add_constraint(constraint=self.constraint)
                loss = torch.cat((loss, loss_pair))
                training_time = torch.cat((training_time, training_time))

            # Reset distance function
            self.distance_fun = distance_fun_original

        if return_loss:
            return loss, training_time
        else:
            return None

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
