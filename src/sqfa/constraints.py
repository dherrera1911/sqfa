"""
Constraints to keep the filters on a certain set (e.g. sphere), or to fix
some filter values during training.
"""

import torch
import torch.nn as nn

__all__ = ["Sphere", "Identity", "FixedFilters"]


def __dir__():
    return __all__


# Define the sphere constraint
class Sphere(nn.Module):
    """Constrains the input tensor to lie on the sphere."""

    def forward(self, X):
        """
        Normalize the input tensor so that it lies on the sphere.

        The norm pooled across channels is computed and used to normalize the tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor in Euclidean space with shape (n_filters, n_dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor lying on the sphere with shape
            (n_filters, n_dim).
        """
        X_normalized = X / X.norm(dim=-1, keepdim=True)
        return X_normalized

    def right_inverse(self, S):
        """
        Identity function to assign to parametrization.

        Parameters
        ----------
        S : torch.Tensor
            Input tensor. Should be different from zero.

        Returns
        -------
        torch.Tensor
            Returns the input tensor `S`.
        """
        return S


# Define the unconstrained constraint
class Identity(nn.Module):
    """Leaves the input tensor unconstrained. Used for consistency."""

    def forward(self, X):
        """
        Return the input tensor as is.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor (n_filters, n_dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor lying on the sphere with shape
            (n_filters, n_dim).
        """
        return X

    def right_inverse(self, S):
        """
        Identity function.

        Parameters
        ----------
        S : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Returns the input tensor.
        """
        return S


class FixedFilters(nn.Module):
    """Fix some of the filters to prevent updating with gradient descent."""

    def __init__(self, n_row_fixed):
        """
        Initialize the FixedFilters class.

        Parameters
        ----------
        value : torch.Tensor
            Value to fix the filters to.
        """
        super().__init__()
        self.n_row_fixed = n_row_fixed

    def forward(self, X):
        """
        Concatenate the fixed tensor with the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Fixed value.
        """
        fixed_tensor = X[: self.n_row_fixed].detach()
        return torch.cat([fixed_tensor, X[self.n_row_fixed :]], dim=0)

    def right_inverse(self, X):
        """
        Return only the rows after the fixed tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Returns the non-fixed part of the tensor.
        """
        return X
