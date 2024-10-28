"""
Distances between Symmetric Positive Definite matrices.
"""

import torch

from .linalg_utils import conjugate_to_identity, spd_sqrt, spd_log, generalized_eigenvalues


__all__ = ["affine_invariant_distance_sq"]


def affine_invariant_distance_sq(A, B):
    # Compute the generalized eigenvalues
    gen_eigvals = generalized_eigenvalues(A, B)
    # Compute the distance
    distance_squared = gs.sum(gs.log(gen_eigvals) ** 2, axis=-1)
    return distance_squared

