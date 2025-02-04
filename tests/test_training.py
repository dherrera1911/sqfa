"""Test functions for training sqfa."""

import pytest
import torch

import sqfa
from make_examples import make_dataset_points, rotated_classes_dataset

MAX_EPOCHS = 50
N_POINTS = 100
N_DIM = 8
torch.manual_seed(1)


def initialize_model(model_type):
    """Generate a tensor of SPD matrices."""
    if model_type == "spd":
        model = sqfa.model.SecondMomentsSQFA(
            n_dim=N_DIM,
            feature_noise=0.001,
            n_filters=2,
        )
    elif model_type == "fisher":
        model = sqfa.model.SQFA(
            n_dim=N_DIM,
            feature_noise=0.001,
            n_filters=2,
        )
    return model


@pytest.mark.parametrize("model_type", ["spd", "fisher"])
def test_training_function(model_type):
    """Test the training function in sqfa._optim."""
    covariances = rotated_classes_dataset()

    model = initialize_model(model_type)

    if model_type == "spd":
        loss, time = sqfa._optim.fitting_loop(
            model=model,
            data_statistics=covariances,
            lr=0.1,
            return_loss=True,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )
    elif model_type == "fisher":
        # Check value error is raised when using only covariances
        with pytest.raises(ValueError):
            loss, time = sqfa._optim.fitting_loop(
                model=model,
                data_statistics=covariances,
                lr=0.1,
                return_loss=True,
                max_epochs=MAX_EPOCHS,
                show_progress=False,
            )
        # Make dictionary with covariance and means input
        stats_dict = {
            "covariances": covariances,
            "means": torch.zeros_like(covariances[:, :, 0]),
        }
        loss, time = sqfa._optim.fitting_loop(
            model=model,
            data_statistics=stats_dict,
            lr=0.1,
            return_loss=True,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"


@pytest.mark.parametrize("feature_noise", [0, 0.001, 0.01])
@pytest.mark.parametrize("n_filters", [1, 2, 4])
@pytest.mark.parametrize("pairwise", [False, True])
def test_training_method(feature_noise, n_filters, pairwise):
    """Test the method `.fit` in the sqfa class."""
    covariances = rotated_classes_dataset()

    model = sqfa.model.SecondMomentsSQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    if n_filters == 1 and pairwise:
        with pytest.raises(ValueError):
            loss, time = model.fit(
                data_statistics=covariances,
                lr=0.1,
                pairwise=pairwise,
                return_loss=True,
                max_epochs=MAX_EPOCHS,
                show_progress=False,
            )
        return
    else:
        loss, time = model.fit(
            data_statistics=covariances,
            lr=0.1,
            pairwise=pairwise,
            return_loss=True,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"


@pytest.mark.parametrize("n_filters", [1, 2, 4])
@pytest.mark.parametrize("feature_noise", [0.001])
def test_pca_init_points(n_filters, feature_noise):
    """Test the method `.fit` in the sqfa class."""
    covariances = rotated_classes_dataset()
    points, labels = make_dataset_points(
        n_points=N_POINTS, class_covariances=covariances
    )

    model = sqfa.model.SecondMomentsSQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    # PCA components
    components = sqfa.statistics.pca(points, n_components=n_filters)
    # PCA initialization
    model.fit_pca(X=points)

    assert torch.allclose(model.filters.detach(), components)

    loss, time = model.fit(
        X=points,
        y=labels,
        lr=0.1,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
        show_progress=False,
    )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"


@pytest.mark.parametrize("n_filters", [1, 2, 4])
@pytest.mark.parametrize("feature_noise", [0.001])
def test_pca_init_scatters(n_filters, feature_noise):
    """Test the method `.fit` in the sqfa class."""
    covariances = rotated_classes_dataset()

    model = sqfa.model.SecondMomentsSQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    # PCA components
    components = sqfa.statistics.pca_from_scatter(covariances, n_components=n_filters)
    # PCA initialization
    model.fit_pca(data_statistics=covariances)

    assert torch.allclose(model.filters.detach(), components)

    loss, time = model.fit(
        data_statistics=covariances,
        lr=0.1,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
        show_progress=False,
    )

    assert loss[-1] is not torch.nan, "Loss is NaN"
    assert not torch.isinf(loss[-1]), "Loss is infinite"
