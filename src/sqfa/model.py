"""Class implementing the Supervised Quadratic Feature Analysis (SQFA) model."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations

from ._optim import fitting_loop
from .constraints import FixedFilters, Identity, Sphere
from .distances import (
    affine_invariant,
    fisher_rao_lower_bound,
)
from .linalg import conjugate_matrix
from .statistics import class_statistics, pca, pca_from_scatter

__all__ = ["SecondMomentsSQFA", "SQFA"]


def __dir__():
    return __all__


def _stats_to_scatter(statistics):
    """
    Convert data_statistics input to scatter matrices. This function
    is used to allow the input to the model to be either
    a dictionary with means and covariances or a tensor with the
    second moments.

    Parameters
    ----------
    statistics : torch.Tensor or dict
        - If a torch.Tensor, should have shape (n_classes, n_dim, n_dim) and contain
          the scatter matrices (second moments) of the data for each class.
        - If a dict, it should contain fields 'means' and 'covariances'.

    Returns
    -------
    torch.Tensor
        Scatter matrices of shape (n_classes, n_dim, n_dim).
    """
    if isinstance(statistics, dict):
        _check_statistics(statistics)

        mean_outer_prod = torch.einsum(
            "ni,nj->nij", statistics["means"], statistics["means"]
        )
        scatter = statistics["covariances"] + mean_outer_prod
    else:
        scatter = statistics

    return scatter


def _check_statistics(data_statistics):
    """
    Check that data_statistics is either:
      1) a torch.Tensor of shape (n_classes, n_dim, n_dim), or
      2) a dictionary containing at least the 'means' and 'covariances' keys.

    Parameters
    ----------
    data_statistics : torch.Tensor or dict
        Data statistics, either as a tensor with second moments or a dictionary
        containing means and covariances.

    Raises
    ------
    ValueError
        If `data_statistics` is a dictionary but does not contain the required keys.
    TypeError
        If `data_statistics` is neither a dictionary nor a tensor-like object.
    """
    if isinstance(data_statistics, dict):
        required_keys = {"means", "covariances"}
        missing_keys = required_keys - set(data_statistics.keys())
        if missing_keys:
            raise ValueError(
                f"`data_statistics` dictionary must contain the keys {required_keys}. "
                f"Missing keys: {missing_keys}"
            )
    elif not hasattr(data_statistics, "shape"):
        # If it's not a dict or something that looks like a tensor, raise a TypeError
        raise TypeError(
            "`data_statistics` must be either a dict with 'means' and 'covariances' "
            "or a torch.Tensor of shape (n_classes, n_dim, n_dim)."
        )


class SecondMomentsSQFA(nn.Module):
    """
    Second-moments Supervised Quadratic Feature Analysis (SQFA) model.
    This version of the model uses only the second moment matrices of the data,
    and distances in the SPD manifold.
    """

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
            self.distance_fun = affine_invariant
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
            Tensor of shape (n_classes, n_dim, n_dim), with second
            moment or covariance matrices.

        Returns
        -------
        torch.Tensor shape (n_classes, n_filters, n_filters)
            Covariances of the transformed features.
        """
        feature_scatters = conjugate_matrix(data_scatters, self.filters)
        return feature_scatters

    def get_class_distances(self, data_statistics, regularized=False):
        """
        Compute the pairwise distances between the feature scatter matrices of the
        different classes.

        Parameters
        ----------
        data_statistics : torch.Tensor or dict
            - If a torch.Tensor, should have shape (n_classes, n_dim, n_dim) and contain
              the scatter matrices (second moments) of the data for each class.
            - If a dict, it should contain fields 'means' and 'covariances'.
        regularized : bool
            If True, regularize the distances by adding a small value to the
            diagonal of the transformed scatter matrices. Default is False.

        Returns
        -------
        torch.Tensor shape (n_classes, n_classes)
            Pairwise distances between the transformed feature scatter matrices.
        """
        # Bring different input type options to the same format
        data_scatters = _stats_to_scatter(data_statistics)

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

    def fit_pca(self, X=None, data_statistics=None):
        """
        Fit the SQFA filters to the data using PCA. This can be used to
        initialize the filters before training.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_dim).
        data_statistics : torch.Tensor
            Tensor of shape (n_classes, n_dim, n_dim) with the second moments
            of the data for each class. If None, then X and y must be provided.
            Default is None.
        """
        if X is None and data_statistics is None:
            raise ValueError("Either X or data_statistics must be provided.")
        if self.filters.shape[0] > self.filters.shape[1]:
            raise ValueError(
                "Number of filters must be less than or equal to the data dimension."
            )

        n_components = self.filters.shape[0]

        if data_statistics is None:
            pca_filters = pca(X, n_components)
        else:
            data_scatters = _stats_to_scatter(data_statistics)
            pca_filters = pca_from_scatter(data_scatters, n_components)

        # Assign fitlers parameter to sqfa
        remove_parametrizations(self, "filters")
        self.filters = nn.Parameter(pca_filters)
        self._add_constraint(constraint=self.constraint)

    def fit(
        self,
        X=None,
        y=None,
        data_statistics=None,
        max_epochs=300,
        lr=0.1,
        estimator="empirical",
        pairwise=False,
        show_progress=True,
        return_loss=False,
        **kwargs,
    ):
        """
        Fit the second-moments SQFA model to data using the LBFGS optimizer.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, n_dim). If data_statistics is None,
            then X and y must be provided.
        y : torch.Tensor
            Labels of shape (n_samples,). If data_statistics is None, then X
            and y must be provided. Labels must be integers starting from 0.
        data_statistics : torch.Tensor or dict
            - If a torch.Tensor, should have shape (n_classes, n_dim, n_dim) and contain
              the scatter matrices (second moments) of the data for each class.
            - If a dict, it should contain fields 'means' and 'covariances'
        max_epochs : int, optional
            Number of max training epochs. By default 50.
        lr : float
            Learning rate for the optimizer. Default is 0.1.
        estimator:
            Covariance estimator to use. Options are "empirical",
            and "oas". Default is "empirical".
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
        if data_statistics is None:
            if X is None or y is None:
                raise ValueError("Either data_statistics or X and y must be provided.")
            data_statistics = class_statistics(X, y, estimator=estimator)

        if not pairwise:
            loss, training_time = fitting_loop(
                model=self,
                data_statistics=data_statistics,
                max_epochs=max_epochs,
                lr=lr,
                show_progress=show_progress,
                return_loss=True,
                **kwargs,
            )

        else:
            n_pairs = self.filters.shape[0] // 2

            # Store initial filters
            filters_original = self.filters.detach().clone()
            noise_original = self.diagonal_noise.detach().clone()[0, 0]

            # Require n_pairs to be even
            if self.filters.shape[0] % 2 != 0:
                raise ValueError(
                    "Number of filters must be even for pairwise training."
                )

            # Loop over pairs
            loss = torch.tensor([])
            training_time = torch.tensor([])
            filters_last_trained = torch.zeros(0)
            for i in range(n_pairs):
                # Re-initialize filters, to be a tensor of shape (2*(i+1), n_dim)
                # with the first 2*i filters being the filters from the previous
                # iteration
                filters_last_trained = self.filters.detach().clone()
                if i == 0:
                    filters_new_init = filters_original[:2]
                else:
                    filters_new_init = torch.cat(
                        (filters_last_trained, filters_original[2 * i : 2 * (i + 1)])
                    )
                remove_parametrizations(self, "filters")
                self.filters = nn.Parameter(filters_new_init)
                self._add_constraint(constraint=self.constraint)

                # Re-initialize noise, to be a tensor of shape (2*(i+1), 2*(i+1))
                feature_noise_mat = noise_original * torch.eye(2 * (i + 1))
                self.register_buffer("diagonal_noise", feature_noise_mat)

                # Fix the filters already trained
                if i > 0:
                    register_parametrization(
                        self, "filters", FixedFilters(n_row_fixed=i * 2)
                    )

                # Train the current pair
                loss_pair, training_time_pair = fitting_loop(
                    model=self,
                    data_statistics=data_statistics,
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
                if training_time.numel() > 0:
                    training_time_pair = training_time_pair + training_time[-1]
                training_time = torch.cat((training_time, training_time_pair))

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


class SQFA(SecondMomentsSQFA):
    """
    Supervised Quadratic Feature Analysis (SQFA) model.
    This version of the model uses both the means and the covariances of the data,
    and uses distances (or approximations) in the manifold of normal distributions.
    """

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
        constraint : str
            Constraint to apply to the filters. Can be 'none', 'sphere' or
            'orthogonal'. Default is 'sphere'.
        """
        if distance_fun is None:
            distance_fun = fisher_rao_lower_bound

        super().__init__(
            n_dim=n_dim,
            feature_noise=feature_noise,
            n_filters=n_filters,
            filters=filters,
            distance_fun=distance_fun,
            constraint=constraint,
        )

    def get_class_distances(self, data_statistics, regularized=False):
        """
        Compute the pairwise lower bounds to the Fisher-Rao distances.

        Parameters
        ----------
        data_statistics : torch.Tensor or dict
            - If a torch.Tensor, should have shape (n_classes, n_dim, n_dim) and contain
              the scatter matrices (second moments) of the data for each class.
            - If a dict, it should contain fields 'means' and 'covariances'.
        regularized : bool
            If True, regularize the distances by adding a small value to the
            diagonal of the transformed scatter matrices. Default is False.

        Returns
        -------
        torch.Tensor shape (n_classes, n_classes)
            Pairwise distances between the transformed feature scatter matrices.
        """
        if not isinstance(data_statistics, dict):
            raise ValueError(
                "data_statistics must be a dictionary with 'means' and 'covariances' keys."
            )
        # Compute feature statistics
        feature_means = self.transform(data_statistics["means"])
        feature_covariances = self.transform_scatters(data_statistics["covariances"])

        if regularized:
            feature_covariances = feature_covariances + self.diagonal_noise[None, :, :]

        feature_statistics = {
          "means": feature_means,
          "covariances": feature_covariances,
        }

        distances = self.distance_fun(
          feature_statistics, feature_statistics
        )
        return distances

    def fit(
        self,
        X=None,
        y=None,
        data_statistics=None,
        max_epochs=300,
        lr=0.1,
        estimator="empirical",
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
            Input data of shape (n_samples, n_dim). If data_statistics is None,
            then X and y must be provided.
        y : torch.Tensor
            Labels of shape (n_samples,). If data_statistics is None, then X
            and y must be provided. Labels must be integers starting from 0.
        data_statistics : dict
            Dictionary containing the fields 'means' and 'covariances'
        max_epochs : int, optional
            Number of max training epochs. By default 50.
        lr : float
            Learning rate for the optimizer. Default is 0.1.
        estimator:
            Covariance estimator to use. Options are "empirical",
            and "oas". Default is "empirical".
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
        if data_statistics is None:
            if X is None or y is None:
                raise ValueError("Either data_statistics or X and y must be provided.")
            data_statistics = class_statistics(X, y, estimator=estimator)
        else:
            if not isinstance(data_statistics, dict):
                raise ValueError(
                    "data_statistics must be a dictionary with 'means' and 'covariances' keys."
                )
            _check_statistics(data_statistics)

        loss, training_time = super().fit(
            X=None,
            y=None,
            data_statistics=data_statistics,
            max_epochs=max_epochs,
            lr=lr,
            estimator=estimator,
            pairwise=pairwise,
            show_progress=show_progress,
            return_loss=True,
            **kwargs,
        )

        if return_loss:
            return loss, training_time
        else:
            return None
