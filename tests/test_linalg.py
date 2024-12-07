"""Tests for the linalg module."""

import pytest
import scipy.linalg
import torch

from make_examples import sample_spd
from sqfa.linalg import (
    conjugate_matrix,
    generalized_eigenvalues,
    generalized_eigenvectors,
    spd_inv_sqrt,
    spd_log,
    spd_sqrt,
)

torch.set_default_dtype(torch.float64)


def generalized_eigenvalues_ref(A, B):
    """
    Compute the generalized eigenvalues of the pair (A, B) using scipy.

    Parameters
    ----------
    A : torch.Tensor
        A tensor of matrices of shape (n_matrices_A, n_dim, n_dim).
    B : torch.Tensor
        A tensor of matrices of shape (n_matrices_B, n_dim, n_dim).

    Returns
    -------
    generalized_eigenvalues : torch.Tensor
        A tensor of generalized eigenvalues of shape (n_matrices_A, n_matrices_B, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    if B.dim() < 3:
        B = B.unsqueeze(0)

    n_matrices_A = A.shape[0]
    n_matrices_B = B.shape[0]
    n_dim = A.shape[1]

    eigvals = torch.zeros((n_matrices_A, n_matrices_B, n_dim))
    for i in range(n_matrices_A):
        for j in range(n_matrices_B):
            eigvals[i, j] = torch.as_tensor(
                scipy.linalg.eigvals(A[i], B[j]),
            ).abs()

    eigvals = torch.sort(eigvals, axis=-1, descending=True)[0]

    return torch.squeeze(eigvals, dim=(0, 1))


def generalized_eigenvectors_ref(A, B):
    """
    Compute the generalized eigenvectors of the pair (A, B) using scipy.

    Parameters
    ----------
    A : torch.Tensor
        A tensor of matrices of shape (n_matrices_A, n_dim, n_dim).
    B : torch.Tensor
        A tensor of matrices of shape (n_matrices_B, n_dim, n_dim).

    Returns
    -------
    generalized_eigenvalues : torch.Tensor
        A tensor of generalized eigenvalues of shape (n_matrices_A, n_matrices_B, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    if B.dim() < 3:
        B = B.unsqueeze(0)

    n_matrices_A = A.shape[0]
    n_matrices_B = B.shape[0]
    n_dim = A.shape[1]

    eigvecs = torch.zeros((n_matrices_A, n_matrices_B, n_dim, n_dim))
    for i in range(n_matrices_A):
        for j in range(n_matrices_B):
            pair_vals, pair_vecs = scipy.linalg.eig(A[i], B[j])
            # Convert to torch tensors
            pair_vals = torch.as_tensor(pair_vals).abs()
            pair_vecs = torch.as_tensor(pair_vecs).abs()
            # Sort the eigenvectors by the eigenvalues in descending order
            _, idx = torch.sort(pair_vals, descending=True)
            # invert the order to get the largest eigenvalues first
            pair_vecs = pair_vecs[:, idx]
            eigvecs[i, j] = pair_vecs

    return torch.squeeze(eigvecs, dim=(0, 1))


def matrix_sqrt_ref(A):
    """
    Compute the square root of a tensor of SPD matrices using scipy.

    Parameters
    ----------
    A : torch.Tensor
        A tensor of SPD matrices of shape (n_matrices, n_dim, n_dim).

    Returns
    -------
    sqrt_A : torch.Tensor
        A tensor of square roots of the input matrices of shape (n_matrices, n_dim, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    n_matrices = A.shape[0]
    n_dim = A.shape[1]
    sqrt_A = torch.zeros((n_matrices, n_dim, n_dim))
    if n_dim > 1:
        for i in range(n_matrices):
            sqrt_A[i] = torch.as_tensor(scipy.linalg.sqrtm(A[i]))
    else:
        sqrt_A = torch.sqrt(A)
    return sqrt_A


def matrix_log_ref(A):
    """
    Compute the matrix logarithm of a tensor of SPD matrices using scipy.

    Parameters
    ----------
    A : torch.Tensor
        A tensor of SPD matrices of shape (n_matrices, n_dim, n_dim).

    Returns
    -------
    log_A : torch.Tensor
        A tensor of matrix logarithms of the input matrices of shape (n_matrices, n_dim, n_dim).
    """
    if A.dim() < 3:
        A = A.unsqueeze(0)
    n_matrices = A.shape[0]
    n_dim = A.shape[1]
    log_A = torch.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        log_A[i] = torch.as_tensor(scipy.linalg.logm(A[i]))

    return log_A


@pytest.fixture(scope="function")
def sample_spd_matrices(n_matrices_A, n_matrices_B, n_dim):
    """Generate tensors of SPD matrices.

    Parameters
    ----------
    n_matrices_A : int
        The number of matrices in the A tensor.
    n_matrices_B : int
        The number of matrices in the B tensor.
    n_dim : int
        The dimension of the matrices.

    Returns
    -------
    A : torch.Tensor
        A tensor of SPD matrices of shape (n_matrices_A, n_dim, n_dim).
    B : torch.Tensor
        A tensor of SPD matrices of shape (n_matrices_B, n_dim, n_dim).
    """
    A = sample_spd(n_matrices_A, n_dim)
    B = sample_spd(n_matrices_B, n_dim)
    return A, B


@pytest.fixture(scope="function")
def sample_filters(n_filters, n_dim):
    """Generate a tensor of random filters."""
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

    # Check the dimensions and shape. The output should have shape (n_matrices_A, n_matrices_B, n_dim),
    # but the batch dimensions n_matrices_A and n_matrices_B are squeezed out if they are length 1
    if n_matrices_A > 1 and n_matrices_B > 1:
        assert eigvals.dim() == 3, (
            "generalized_eigenvalues() output does not have the correct number"
            "of dimensions for A.dim()>1 and B.dim()>1."
        )
        assert (
            eigvals.shape[1] == n_matrices_B
        ), "generalized_eigenvalues() does not match B tensor shape"
        assert (
            eigvals.shape[0] == n_matrices_A
        ), "generalized_eigenvalues() output does not match A tensor shape"
        assert eigvals.shape[-1] == n_dim, (
            "generalized_eigenvalues() output does not match the dimension"
            "of the matrices."
        )
    elif n_matrices_A == 1 and n_matrices_B == 1:
        assert eigvals.dim() == 1, (
            "generalized_eigenvalues() output does not have the correct number"
            "of dimensions for A.dim()==1 and B.dim()==1."
        )
        assert (
            eigvals.shape[-1] == n_dim
        ), "generalized_eigenvalues() output does not match the dimension of the matrices."
    elif n_matrices_A == 1:
        assert eigvals.dim() == 2, (
            "generalized_eigenvalues() output does not have the correct number"
            "of dimensions for A.dim()==1 and B.dim()>1."
        )
        assert (
            eigvals.shape[0] == n_matrices_B
        ), "generalized_eigenvalues() output does not match B tensor shape"
        assert eigvals.shape[-1] == n_dim, (
            "generalized_eigenvalues() output does not match the dimension "
            "of the matrices."
        )
    elif n_matrices_B == 1:
        assert eigvals.dim() == 2, (
            "generalized_eigenvalues() output has incorrect number of dimensions"
            "for A.dim()>1 and B.dim()==1."
        )
        assert (
            eigvals.shape[0] == n_matrices_A
        ), "generalized_eigenvalues() output does not match A tensor shape"
        assert (
            eigvals.shape[-1] == n_dim
        ), "generalized_eigenvalues() has incorrect dimensions for B.dim()==1."

    reference_eigvals = generalized_eigenvalues_ref(A, B)
    reference_eigvals = reference_eigvals
    assert torch.allclose(
        eigvals, reference_eigvals, atol=1e-5
    ), "Generalized eigenvalues are not correct."


@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1, 4, 8])
@pytest.mark.parametrize("n_dim", [2, 4, 6])
def test_generalized_eigenvectors(
    sample_spd_matrices, n_matrices_A, n_matrices_B, n_dim
):
    """Test the generalized eigenvalues function."""
    A, B = sample_spd_matrices
    eigvecs, eigvals = generalized_eigenvectors(A, B)
    eigvecs_ref = generalized_eigenvectors_ref(A, B)
    eigvecs_ref = eigvecs_ref
    assert torch.allclose(
        torch.abs(eigvecs), torch.abs(eigvecs_ref), atol=1e-4
    ), "Generalized eigenvectors are not correct."


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
        assert (
            filter_conjugate.shape[0] == n_matrices_A
        ), "Conjugate f A f^T does not match batch dimension in A."
        assert (
            filter_conjugate.shape[-1] == n_filters
        ), "Conjugate f A f^T does not match filter dimension."
        assert (
            filter_conjugate.dim() == 3
        ), "Conjugate f A f^T has incorrect number of dimensions for A.dim()>1."
    else:
        assert filter_conjugate.dim() == 2, (
            "Conjugate f A f^T does not have the correct number of dimensions "
            "for A.dim()==1."
        )
        assert (
            filter_conjugate.shape[-1] == n_filters
        ), "Conjugate f A f^T does not match filter dimension for A.dim()==1."

    A_ref = A.unsqueeze(0) if A.dim() < 3 else A

    # Apply filters to conjugate A
    filter_conjugate_ref = torch.einsum("ij,kjl,lm->kim", filters, A_ref, filters.T)
    filter_conjugate_ref = torch.squeeze(filter_conjugate_ref, dim=0)

    assert torch.allclose(
        filter_conjugate, filter_conjugate_ref, atol=1e-5
    ), "Conjugate f A f^T is not correct."


@pytest.mark.parametrize("n_dim", [2, 4])
@pytest.mark.parametrize("n_matrices_A", [1, 4])
@pytest.mark.parametrize("n_matrices_B", [1])
def test_spd_inv_sqrt(sample_spd_matrices, n_dim, n_matrices_A, n_matrices_B):
    """Test the conjugate_matrix function."""
    A, B = sample_spd_matrices

    A_inv_sqrt = spd_inv_sqrt(A)

    A_conjugate = conjugate_matrix(A, A_inv_sqrt)

    identity = torch.eye(n_dim)
    if n_matrices_A > 1:
        i = torch.arange(0, n_matrices_A)
        A_conjugate = A_conjugate[i, i]
        identity = identity.unsqueeze(0).repeat(n_matrices_A, 1, 1)

    assert torch.allclose(
        A_conjugate, identity, atol=1e-5
    ), "Inverse square root is not correct."


@pytest.mark.parametrize("n_dim", [2, 4, 17])
@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1])
def test_spd_sqrt(sample_spd_matrices, n_matrices_A, n_matrices_B, n_dim):
    """Test the spd matrix square root function."""
    A, B = sample_spd_matrices
    sqrt_A = spd_sqrt(A)
    sqrt_A_ref = matrix_sqrt_ref(A)
    sqrt_A_ref = torch.as_tensor(sqrt_A_ref)

    if n_matrices_A > 1:
        assert (
            sqrt_A.dim() == 3
        ), "spd_sqrt() output has incorrect number of dimensions for A.dim()>1."
        assert (
            sqrt_A.shape[-1] == n_dim
        ), "spd_sqrt() output does not match the dimension of the matrices."
    else:
        assert (
            sqrt_A.dim() == 2
        ), "spd_sqrt() output has inccorrect number of dimensions for A.dim()==1."
        assert (
            sqrt_A.shape[-1] == n_dim
        ), "spd_sqrt() output does not match the dimension of the matrices."

    assert torch.allclose(
        sqrt_A, sqrt_A_ref, atol=1e-4
    ), "SPD square root is not correct."


@pytest.mark.parametrize("n_dim", [2, 4, 17])
@pytest.mark.parametrize("n_matrices_A", [1, 4, 8])
@pytest.mark.parametrize("n_matrices_B", [1])
def test_spd_log(sample_spd_matrices, n_matrices_A, n_matrices_B, n_dim):
    """Test the spd matrix square root function."""
    A, B = sample_spd_matrices
    log_A = spd_log(A)
    log_A_ref = matrix_log_ref(A)
    log_A_ref = torch.as_tensor(log_A_ref)

    if n_matrices_A > 1:
        assert (
            log_A.dim() == 3
        ), "spd_log() output has incorrect number of dimensions for A.dim()>1."
        assert (
            log_A.shape[-1] == n_dim
        ), "spd_log() output does not match the dimension of the matrices."
    else:
        assert (
            log_A.dim() == 2
        ), "spd_log() output has incorrect number of dimensions for A.dim()==1."
        assert (
            log_A.shape[-1] == n_dim
        ), "spd_log() output does not match the dimension of the matrices."

    assert torch.allclose(log_A, log_A_ref, atol=1e-4), "SPD log is not correct."
