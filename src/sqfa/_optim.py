"""Routine to fit SQFA filters using Gradient Descent."""

import time

import torch
from torch import optim
from tqdm import tqdm

__all__ = ["fitting_loop"]


def __dir__():
    return __all__


def check_distances_valid(distances):
    """
    Check if off-diagonal distances are valid. Raise an error if they are not.

    Parameters
    ----------
    distances : torch.Tensor
        Tensor of pairwise distances between covariance matrices.
    """
    n_classes = distances.shape[0]
    tril_ind = torch.tril_indices(n_classes, n_classes, offset=-1)
    if torch.isnan(distances[tril_ind]).any():
        raise ValueError("Some distances between classes are NaN.")
    if torch.isinf(distances[tril_ind]).any():
        raise ValueError("Some distances between classes are inf.")


def fitting_loop(
    model,
    data_scatters,
    max_epochs=200,
    lr=0.1,
    atol=1e-6,
    show_progress=True,
    return_loss=False,
    **kwargs,
):
    """
    Learn SQFA filters using LBFGS optimizer.

    Parameters
    ----------
    model : SQFA model object
        The model used for fitting.
    data_scatters : torch.Tensor
        Tensor of shape (n_classes, n_dim, n_dim) with the scatter matrices
        of the data for each class.
    distance_fun : callable
        Function returning pairwise distances between covariance matrices.
        Takes as input two batches of covariance matrices of shape (batch_size, n_dim, n_dim)
        and return a tensor of shape (batch_size, batch_size).
    max_epochs : int, optional
        Number of max training epochs. By default 50.
    lr : float, optional
        Learning rate, by default 0.1.
    atol : float, optional
        Tolerance for stopping training, by default 1e-8.
    show_progress : bool
        If True, show a progress bar during training. Default is True.
    return_loss : bool
        If True, return the loss after training. Default is False.
    kwargs : dict
        Additional arguments to pass to LBFGS optimizer.

    Returns
    -------
    torch.Tensor
        Tensor containing the loss at each epoch (shape: epochs).
    torch.Tensor
        Tensor containing the training time at each epoch (shape: epochs).
    """
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        **kwargs,
    )

    n_classes = data_scatters.shape[0]
    tril_ind = torch.tril_indices(n_classes, n_classes, offset=-1)

    def closure():
        optimizer.zero_grad()
        distances = model.get_class_distances(data_scatters, regularized=True)
        check_distances_valid(distances)
        epoch_loss = -torch.mean(distances[tril_ind[0], tril_ind[1]])
        epoch_loss.backward()
        return epoch_loss

    loss_list = []
    training_time = []
    total_start_time = time.time()

    prev_loss = 0.0
    consecutive_stopping_criteria_met = 0

    for e in tqdm(
        range(max_epochs), desc="Epochs", unit="epoch", disable=not show_progress
    ):
        epoch_loss = optimizer.step(closure)
        epoch_time = time.time() - total_start_time

        loss_change = abs(prev_loss - epoch_loss.item())

        # Update tqdm bar description with loss change and total time
        # tqdm.write(
        #    f"Epoch {e+1}/{epochs}, Loss: {epoch_loss.item():.4f}, "
        #    f"Change: {loss_change:.4f}, Time: {epoch_time:.2f}s"
        # )

        # Check if loss change is below atol
        if loss_change < atol:
            consecutive_stopping_criteria_met += 1
        else:
            consecutive_stopping_criteria_met = 0

        prev_loss = epoch_loss.item()
        training_time.append(epoch_time)
        loss_list.append(epoch_loss.item())

        # Stop if loss change is below atol for 3 consecutive epochs
        if consecutive_stopping_criteria_met >= 3:
            tqdm.write(
                f"Loss change below {atol} for 3 consecutive epochs. Stopping training at epoch {e + 1}/{max_epochs}."
            )
            break

    else:  # Executes if no break occurs
        print(
            f"Reached max_epochs ({max_epochs}) without meeting stopping criteria."
            + "Consider increasing max_epochs, changing initialization or using dtype=torch.float64."
        )

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None
