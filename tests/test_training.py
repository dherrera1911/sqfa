"""Test functions for training sqfa."""

import numpy as np
import pytest
import torch

import sqfa

MAX_EPOCHS = 20

@pytest.fixture(scope="function")
def disparity_covariances():
    """Load disparity covariances from file."""
    # Load covariances
    covariances = torch.as_tensor(
        np.loadtxt("tests/test_data/disparity_covariances_noisy.csv", delimiter=",")
    )
    # Reshape to covariance shape
    covariances = torch.reshape(covariances.T, (19, 52, 52))
    covariances = covariances.float()
    return covariances


def test_training_function(disparity_covariances):
    """Test the training function in sqfa._optim."""
    covariances = disparity_covariances

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=0.01,
        n_filters=2,
    )

    loss, time = sqfa._optim.fitting_loop(
        model=model,
        second_moments=covariances,
        lr=0.1,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
    )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
    assert loss[-1] < loss[0], "Loss did not decrease"


@pytest.mark.parametrize("feature_noise", [0, 0.01, 0.1])
@pytest.mark.parametrize("n_filters", [1, 2, 6])
@pytest.mark.parametrize("pairwise", [False, True])
def test_training_method(disparity_covariances, feature_noise, n_filters, pairwise):
    """Test the method `.fit` in the sqfa class."""
    covariances = disparity_covariances

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    if n_filters == 1 and pairwise:
        with pytest.raises(ValueError):
            loss, time = model.fit(
                second_moments=covariances,
                lr=0.1,
                pairwise=pairwise,
                return_loss=True,
                max_epochs=MAX_EPOCHS,
            )
        return
    else:
        loss, time = model.fit(
            second_moments=covariances,
            lr=0.1,
            pairwise=pairwise,
            return_loss=True,
            max_epochs=MAX_EPOCHS,
        )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
    assert loss[-1] < loss[0] or len(loss) == 1, "Loss did not decrease"
