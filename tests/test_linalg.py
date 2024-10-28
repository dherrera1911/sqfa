"""Tests for the linalg_utils module."""
import numpy as np
import pytest
import scipy.linalg
import torch

from make_examples import sample_spd
from sqfa.linalg_utils import (
    conjugate_matrix,
    generalized_eigenvalues,
)


def generalized_eigenvalues_ref(A, B):
    """
    Compute the generalized eigenvalues of the pair (A, B) using scipy.

    Parameters
    ----------
    A : np.ndarray
        A tensor of matrices of shape (n_matrices_A, n_dim, n_dim).
    B : np.ndarray
        A tensor of matrices of shape (n_matrices_B, n_dim, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    if B.dim() < 3:
        B = B.unsqueeze(0)

    n_matrices_A = A.shape[0]
    n_matrices_B = B.shape[0]
    n_dim = A.shape[1]

    generalized_eigenvalues = np.zeros((n_matrices_A, n_matrices_B, n_dim))
    for i in range(n_matrices_A):
        for j in range(n_matrices_B):
            generalized_eigenvalues[i, j] = scipy.linalg.eigvals(A[i], B[j])

    generalized_eigenvalues = torch.as_tensor(
        np.sort(generalized_eigenvalues, axis=-1), dtype=torch.float32
    )

    return torch.squeeze(generalized_eigenvalues)


@pytest.fixture(scope="function")
def sample_spd_matrices(n_matrices_A, n_matrices_B, n_dim):
    """Generate a tensor of SPD matrices."""
    A = sample_spd(n_matrices_A, n_dim)
    B = sample_spd(n_matrices_B, n_dim)
    return A, B


@pytest.fixture(scope="function")
def sample_filters(n_filters, n_dim):
    """Generate a tensor of filters."""
    filters = torch.randn(n_filters, n_dim)
    return filters


@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_generalized_eigenvalues(
    sample_spd_matrices, n_matrices_A, n_matrices_B, n_dim
):
    """Test the generalized eigenvalues function."""
    A, B = sample_spd_matrices
    eigvals = generalized_eigenvalues(A, B)

    if n_matrices_A > 1 and n_matrices_B > 1:
        assert eigvals.dim() == 3, (
            "The output does not have the correct number of dimensions for "
            "A.dim()>1 and B.dim()>1."
        )
        assert (
            eigvals.shape[1] == n_matrices_B
        ), "The output does not match B tensor shape"
        assert (
            eigvals.shape[0] == n_matrices_A
        ), "The output does not match A tensor shape"
        assert (
            eigvals.shape[-1] == n_dim
        ), "The output does not match the dimension of the matrices."
    elif n_matrices_A == 1 and n_matrices_B == 1:
        assert eigvals.dim() == 1, (
            "The output does not have the correct number of dimensions for "
            "A.dim()==1 and B.dim()==1."
        )
        assert (
            eigvals.shape[-1] == n_dim
        ), "The output does not match the dimension of the matrices."
    elif n_matrices_A == 1:
        assert eigvals.dim() == 2, (
            "The output does not have the correct number of dimensions for "
            "A.dim()==1 and B.dim()>1."
        )
        assert (
            eigvals.shape[0] == n_matrices_B
        ), "The output does not match B tensor shape"
        assert (
            eigvals.shape[-1] == n_dim
        ), "The output does not match the dimension of the matrices."
    elif n_matrices_B == 1:
        assert eigvals.dim() == 2, (
            "The output does not have the correct number of dimensions for A.dim()>1 "
            "and B.dim()==1."
        )
        assert (
            eigvals.shape[0] == n_matrices_A
        ), "The output does not match A tensor shape"
        assert (
            eigvals.shape[-1] == n_dim
        ), "The output does not match the dimension of the matrices."

    reference_eigvals = generalized_eigenvalues_ref(A, B)
    assert torch.allclose(
        eigvals, reference_eigvals
    ), "Generalized eigenvalues are not correct."


@pytest.mark.parametrize("n_dim", [2, 4, 6])
@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1])
@pytest.mark.parametrize("n_filters", [1, 4, 7])
def test_conjugate_matrix(
    sample_spd_matrices, sample_filters, n_dim, n_matrices_A, n_matrices_B, n_filters
):
    """Test the conjugate_matrix function."""
    filters = sample_filters
    A, B = sample_spd_matrices

    if n_filters == 1:
        with pytest.raises(ValueError, match="B must have at least 2 dimensions"):
            filter_conjugate = conjugate_matrix(A, torch.squeeze(filters))

    filter_conjugate = conjugate_matrix(A, filters)

    # Check the dimensions and shape of the output
    if n_matrices_A > 1:
        if n_filters > 1:
            assert (
                filter_conjugate.shape[0] == n_matrices_A
            ), "Conjugate f A f^T does not match batch dimension in A."
            assert (
                filter_conjugate.shape[-1] == n_filters
            ), "Conjugate f A f^T does not match filter dimension."
        else:
            assert (
                filter_conjugate.shape[0] == n_matrices_A
            ), "Conjugate f A f^T does not match batch dimension in A for f.dim()=1."
            assert filter_conjugate.dim() == 1, (
                "Conjugate f A f^T does not have the correct number of "
                "dimensions for f.dim()==1."
            )
    else:
        if n_filters > 1:
            assert filter_conjugate.dim() == 2, (
                "Conjugate f A f^T does not have the correct number of dimensions "
                "for A.dim()==1."
            )
            assert (
                filter_conjugate.shape[-1] == n_filters
            ), "Conjugate f A f^T does not match filter dimension for A.dim()==1."
        else:
            assert filter_conjugate.dim() == 0, (
                "Conjugate f A f^T does not have the correct number of dimensions for "
                "A.dim()==1 and f.dim()==1 case."
            )

    A_ref = A.unsqueeze(0) if A.dim() < 3 else A

    # Apply filters to conjugate A
    filter_conjugate_ref = torch.einsum("ij,kjl,lm->kim", filters, A_ref, filters.T)
    filter_conjugate_ref = torch.squeeze(filter_conjugate_ref)

    assert torch.allclose(
        filter_conjugate, filter_conjugate_ref
    ), "Conjugate f A f^T is not correct."
