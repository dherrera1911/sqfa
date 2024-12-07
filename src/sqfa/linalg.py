"""Utility functions for matrix algebra."""

import torch

__all__ = [
    "conjugate_matrix",
    "generalized_eigenvalues",
    "generalized_eigenvectors",
    "spd_sqrt",
    "spd_log",
    "spd_inv_sqrt",
]


def __dir__():
    return __all__


def conjugate_matrix(A, B):
    """
    Conjugate matrix A by B, i.e. compute B A B^T.

    Parameters
    ----------
    A : torch.Tensor
        Matrix A. Shape (n_batch_A, n_dim, n_dim).
    B : torch.Tensor
        Matrix B. Shape (n_batch_B, n_dim, n_out).

    Returns
    -------
    C : torch.Tensor
        The conjugated matrix. Shape (n_batch_A, n_batch_B, n_out, n_out).
        If a batch dimension is 1, it is squeezed out.
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() < 2:
        raise ValueError("B must have at least 2 dimensions.")
    # Use einsum
    C = torch.einsum("...ij,njk,...kl->n...il", B, A, B.transpose(-2, -1))
    # Use matmul
    # C = B[None, ...] @ A[:, None, ...] @ B.transpose(-2, -1)[None, ...]
    squeeze_dim = (0) if B.dim() == 2 else (0, 1)
    return torch.squeeze(C, dim=squeeze_dim)


def generalized_eigenvalues(A, B):
    """
    Compute the generalized eigenvalues of the pair of symmetric positive
    definite matrices (A, B).

    Parameters
    ----------
    A : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch_A, n_dim, n_dim).
    B : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch_B, n_dim, n_dim).

    Returns
    -------
    eigenvalues : torch.Tensor
        The generalized eigenvalues of the pair (A, B), sorted in descending
        order. Shape (n_batch_A, n_batch_B, n_dim).
        If a batch dimension is 1, it is squeezed out.
    """
    B_inv_sqrt = spd_inv_sqrt(B)
    A_conj = conjugate_matrix(A, B_inv_sqrt)
    eigenvalues = torch.linalg.eigvalsh(A_conj)
    return eigenvalues.flip(-1)


def generalized_eigenvectors(A, B):
    """
    Compute the generalized eigenvectors of the pair of symmetric positive
    definite matrices (A, B).

    Parameters
    ----------
    A : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch_A, n_dim, n_dim).
    B : torch.Tensor
        Symmetric positive definite matrix. Shape (n_batch_B, n_dim, n_dim).

    Returns
    -------
    eigenvectors: torch.Tensor
        The generalized eigenvectors sorted in descending order
        of the eigenvalues.
        Shape (n_batch_A, n_batch_B, n_dim, n_dim).
        If a batch dimension is 1, it is squeezed out.
    eigenvalues : torch.Tensor
        The generalized eigenvalues sorted in descending order.
        Shape (n_batch_A, n_batch_B, n_dim).
        If a batch dimension is 1, it is squeezed out.
    """
    B_inv_sqrt = spd_inv_sqrt(B)
    A_conj = conjugate_matrix(A, B_inv_sqrt)
    if A.dim() == 2:
        A_conj = A_conj.unsqueeze(0)
    if B.dim() == 2:
        B_inv_sqrt = B_inv_sqrt.unsqueeze(0)
        A_conj = A_conj.unsqueeze(1)
    eigenvalues, eigenvectors = torch.linalg.eigh(A_conj)

    # Flip the order of the eigenvectors
    eigenvectors = eigenvectors.flip(-1)
    eigenvalues = eigenvalues.flip(-1)

    # Transform eigenvectors back to the original basis
    eigenvectors = torch.einsum(
        "bij,abjk->abik", B_inv_sqrt.transpose(-2, -1), eigenvectors
    )
    eigenvectors = eigenvectors / torch.linalg.norm(eigenvectors, dim=-2, keepdim=True)

    return torch.squeeze(eigenvectors, dim=(0, 1)), torch.squeeze(
        eigenvalues, dim=(0, 1)
    )


def spd_sqrt(M):
    """
    Compute the square root of a symmetric positive definite matrix.

    Computes the symmetric positive definite matrix S such that SS = M.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (..., n_dim, n_dim).

    Returns
    -------
    M_sqrt : torch.Tensor
        The square root of M. Shape (..., n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    M_sqrt = torch.einsum(
        "...ij,...j,...kj->...ik", eigvecs, torch.sqrt(eigvals), eigvecs
    )
    return M_sqrt


def spd_inv_sqrt(M):
    """
    For symmetric positive definite matrix M, compute the inverse square
    root of M.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (n_batch, n_dim, n_dim).

    Returns
    -------
    M_inv_sqrt : torch.Tensor
        Inverse square root of M. Shape (n_batch, n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    inv_sqrt_eigvals = torch.sqrt(1.0 / eigvals)
    M_inv_sqrt = eigvecs * inv_sqrt_eigvals.unsqueeze(-2)
    return M_inv_sqrt.transpose(-2, -1)


def spd_log(M):
    """
    Compute the matrix logarithm of a symmetric positive definite matrix.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive definite matrices. Shape (..., n_dim, n_dim).

    Returns
    -------
    M_log : torch.Tensor
        The matrix logarithm of M. Shape (..., n_dim, n_dim).
    """
    eigvals, eigvecs = torch.linalg.eigh(M)
    M_log = torch.einsum(
        "...ij,...j,...kj->...ik", eigvecs, torch.log(eigvals), eigvecs
    )
    return M_log
