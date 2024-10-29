"""Utility functions for matrix algebra."""

import torch

__all__ = ["conjugate_matrix", "generalized_eigenvalues", "spd_sqrt", "spd_log"]


def __dir__():
    return __all__


def conjugate_matrix(A, B):
    """
    Conjugate matrix A by B, i.e. compute B A B^T.

    ----------
    A : torch.Tensor
        Matrix A. Shape (n_batch, n_dim, n_dim).
    B : torch.Tensor
        Matrix B. Shape (..., n_dim, n_out).

    Returns
    -------
    C : torch.Tensor
        The conjugated matrix. Shape (..., n_batch, n_out, n_out).
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() < 2:
        raise ValueError("B must have at least 2 dimensions.")
    # Use einsum
    C = torch.einsum("...ij,njk,...kl->n...il", B, A, B.transpose(-2, -1))
    # Use matmul
    # C = torch.matmul(B.transpose(-2,-1), torch.matmul(A, B))
    return torch.squeeze(C)


def conjugate_to_identity(M):
    """
    For symmetric positive definite matrix M, compute the matrix C such that
    C M C^T = I, where I is the identity matrix.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (n_batch, n_dim, n_dim).

    Returns
    -------
    C : torch.Tensor
        The matrix C such that C M C^T = I. Shape (n_batch, n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_sqrt_eigvals = torch.sqrt(1.0 / eigvals)
    C = eigvecs * inv_sqrt_eigvals.unsqueeze(-2)
    return C.transpose(-2, -1)


def generalized_eigenvalues(A, B):
    """
    Compute the generalized eigenvalues of the pair of symmetric positive
    definite matrices (A, B).

    Parameters
    ----------
    A : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch, n_dim, n_dim).
    B : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch, n_dim, n_dim).

    Returns
    -------
    eigenvalues : torch.Tensor
        The generalized eigenvalues of the pair (A, B). Shape (n_batch, n_dim).
    """
    C = conjugate_to_identity(B)
    A_conj = conjugate_matrix(A, C)
    eigenvalues = torch.linalg.eigvalsh(A_conj)
    return eigenvalues


def spd_sqrt(M):
    """
    Compute the square root of a symmetric positive definite matrix.

    Computes the symmetric positive definite matrix S such that SS = M.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (n_batch, n_dim, n_dim).

    Returns
    -------
    M_sqrt : torch.Tensor
        The square root of M. Shape (n_batch, n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    M_sqrt = torch.einsum(
        "...ij,...j,...kj->...ik", eigvecs, torch.sqrt(eigvals), eigvecs
    )
    return M_sqrt


def spd_log(M):
    """
    Compute the matrix logarithm of a symmetric positive definite matrix.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (n_batch, n_dim, n_dim).

    Returns
    -------
    M_log : torch.Tensor
        The matrix logarithm of M. Shape (n_batch, n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    M_log = torch.einsum(
        "...ij,...j,...kj->...ik", eigvecs, torch.log(eigvals), eigvecs
    )
    return M_log
