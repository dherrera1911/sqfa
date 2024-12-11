"""Test functions for training sqfa."""

import pytest
import torch

import sqfa
from make_examples import rotated_classes_dataset

MAX_EPOCHS = 50
MAX_TRIES = 3
torch.manual_seed(0)


@pytest.fixture(scope="function")
def make_dataset():
    """Create a dataset of 8 classes with rotated covariances."""
    class_covariances = rotated_classes_dataset()
    return class_covariances


def test_training_function(make_dataset):
    """Test the training function in sqfa._optim."""
    covariances = make_dataset

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=0.001,
        n_filters=2,
    )

    loss, time = sqfa._optim.fitting_loop(
        model=model,
        data_scatters=covariances,
        lr=0.1,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
        show_progress=False,
    )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
    assert loss[-1] < loss[0], "Loss did not decrease"


@pytest.mark.parametrize("feature_noise", [0, 0.001, 0.01])
@pytest.mark.parametrize("n_filters", [1, 2, 4])
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
                show_progress=False,
            )
        return
    else:
        n_tries = 0
        loss_decreased = False
        while n_tries < MAX_TRIES and not loss_decreased:
            loss, time = model.fit(
                data_scatters=covariances,
                lr=0.1,
                pairwise=pairwise,
                return_loss=True,
                max_epochs=MAX_EPOCHS,
                show_progress=False,
            )
            n_tries += 1
            loss_decreased = loss[-1] < loss[0]

    assert loss[-1] < loss[0] or len(loss) == 1, "Loss did not decrease in 3 tries"
    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
