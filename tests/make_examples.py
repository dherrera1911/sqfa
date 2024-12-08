"""Generate synthetic data for testing."""

import torch


def make_orthogonal_matrices(n_matrices, n_dim):
    """Generate random orthogonal matrices."""
    low_tri = torch.randn(n_matrices, n_dim, n_dim)
    low_tri = torch.tril(low_tri, diagonal=-1)
    skew_sym = low_tri - low_tri.transpose(1, 2)
    orthogonal = torch.matrix_exp(skew_sym)
    return orthogonal


def sample_spd(n_matrices, n_dim):
    """Generate random SPD matrices."""
    eigvals = 2 * (torch.rand(n_matrices, n_dim)) ** 2 + 0.01
    eigvecs = make_orthogonal_matrices(n_matrices, n_dim)
    spd = torch.einsum("ijk,ik,ikl->ijl", eigvecs, eigvals, eigvecs.transpose(1, 2))
    return torch.squeeze(spd)


def make_rotation_matrix(theta, dims, n_dim):
    """Make a matrix that rotates 2 dimensions of a 6x6 matrix by theta.

    Args:
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    theta = torch.deg2rad(theta)
    rotation = torch.eye(n_dim)
    rot_mat_2 = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )
    for row in range(2):
        for col in range(2):
            rotation[dims[row], dims[col]] = rot_mat_2[row, col]
    return rotation


def rotate_classes(base_cov, angles, dims):
    """Rotate 2 dimensions of base_cov, specified in dims, by the angles in the angles list
    Args:
        base_cov (torch.Tensor): Base covariances
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    if len(angles) != base_cov.shape[0]:
        raise ValueError("The number of angles must be equal to the number of classes.")

    n_dim = base_cov.shape[-1]

    for i, theta in enumerate(angles):
        rotation_matrix = make_rotation_matrix(theta, dims, n_dim)
        base_cov[i] = torch.einsum(
            "ij,jk,kl->il", rotation_matrix, base_cov[i], rotation_matrix.T
        )
    return base_cov


def rotated_classes_dataset():
    """Create a dataset of 4 classes with rotated covariances in 8 dimensions."""
    angle_base = torch.tensor([0, 1, 2, 3, 4])
    angles = [
        angle_base * 20,  # Dimensions 1, 2
        angle_base * 10,  # Dimensions 3, 4
        angle_base * 5,  # Dimensions 5, 6
        angle_base * 2,  # Dimensions 7, 8
    ]

    n_classes = len(angles[0])
    variances = torch.tensor([1.00, 0.04, 1.0, 0.04, 1.00, 0.04, 1.00, 0.04])
    base_cov = torch.diag(variances)
    base_cov = base_cov.repeat(n_classes, 1, 1)

    class_covariances = base_cov
    for d in range(len(angles)):
        ang = torch.as_tensor(angles[d])
        class_covariances = rotate_classes(
            class_covariances, ang, dims=[2 * d, 2 * d + 1]
        )
    return class_covariances
