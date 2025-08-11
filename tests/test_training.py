"""Test functions for training sqfa."""

import pytest
import torch

import sqfa
from make_examples import make_dataset_points, rotated_classes_dataset

MAX_EPOCHS = 40
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
        with pytest.raises(TypeError):
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
        # Error that pairwise needs even filters
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

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=feature_noise,
        n_filters=n_filters,
    )

    data_stats = {
      'means': torch.zeros_like(covariances[:, :, 0]),
      'covariances': covariances
    }

    loss, time = model.fit(
        data_statistics=data_stats,
        lr=0.1,
        pairwise=pairwise,
        return_loss=True,
        max_epochs=MAX_EPOCHS,
        show_progress=False,
    )


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

    with pytest.raises(ValueError):
        model.fit_pca()


@pytest.mark.parametrize("pairwise", [False, True])
def test_input_check(pairwise):
    """Test that when wrong input is given, error is raised."""
    covariances = rotated_classes_dataset()

    model = sqfa.model.SQFA(
        n_dim=covariances.shape[-1],
        feature_noise=0,
        n_filters=2,
    )

    # Not providing a dict for SQFA
    with pytest.raises(TypeError):
        model.fit(
            data_statistics=covariances,
            lr=0.1,
            pairwise=pairwise,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )

    # Not providing any data
    with pytest.raises(ValueError):
        model.fit(
            lr=0.1,
            pairwise=pairwise,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )

    model = sqfa.model.SecondMomentsSQFA(
        n_dim=covariances.shape[-1],
        feature_noise=0,
        n_filters=2,
    )

    # Providing list instead of tensor or dict
    cov_list = [covariances[i] for i in range(covariances.shape[0])]
    with pytest.raises(TypeError):
        model.fit(
            data_statistics=cov_list,
            lr=0.1,
            pairwise=pairwise,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )


    with pytest.raises(ValueError):
        model.fit(
            lr=0.1,
            pairwise=pairwise,
            max_epochs=MAX_EPOCHS,
            show_progress=False,
        )

