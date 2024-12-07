"""Test functions for training sqfa."""

import pytest
import torch

import sqfa

MAX_EPOCHS = 100
torch.manual_seed(0)


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


def make_rotated_classes(base_cov, angles, dims):
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


@pytest.fixture(scope="function")
def make_dataset():
    """Create a dataset of 8 classes with rotated covariances."""
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
        ang = torch.tensor(angles[d])
        class_covariances = make_rotated_classes(
            class_covariances, ang, dims=[2 * d, 2 * d + 1]
        )
    return class_covariances


def test_training_function(make_dataset):
    """Test the training function in sqfa._optim."""
    covariances = make_dataset

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=0.01,
        n_filters=2,
    )

    loss, time = sqfa._optim.fitting_loop(
        model=model,
        data_scatters=covariances,
        lr=0.1,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
    )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
    assert loss[-1] < loss[0], "Loss did not decrease"


@pytest.mark.parametrize("feature_noise", [0, 0.001, 0.01])
@pytest.mark.parametrize("n_filters", [1, 2, 6])
@pytest.mark.parametrize("pairwise", [False, True])
def test_training_method(make_dataset, feature_noise, n_filters, pairwise):
    """Test the method `.fit` in the sqfa class."""
    covariances = make_dataset

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    if n_filters == 1 and pairwise:
        with pytest.raises(ValueError):
            loss, time = model.fit(
                data_scatters=covariances,
                lr=0.1,
                pairwise=pairwise,
                return_loss=True,
                max_epochs=MAX_EPOCHS,
            )
        return
    else:
        loss, time = model.fit(
            data_scatters=covariances,
            lr=0.1,
            pairwise=pairwise,
            return_loss=True,
            max_epochs=MAX_EPOCHS,
        )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
    assert loss[-1] < loss[0] or len(loss) == 1, "Loss did not decrease"
