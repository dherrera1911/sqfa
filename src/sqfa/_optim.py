"""
Routine to fit SQFA filters using Gradient Descent.
"""

import time

import torch
from torch import optim
from tqdm import tqdm

from .distances import ai_distance_sq

__all__ = ["fitting_loop"]


def __dir__():
    return __all__


def fitting_loop(
    model,
    distance_fun,
    epochs=30,
    learning_rate=0.1,
    decay_step=1000,
    decay_rate=1,
):
    """
    Learn SQFA filters using Gradient Descent.

    Parameters
    ----------
    model : SQFA model object
        The model used for fitting.
    distance_fun : callable
        Function returning pairwise distances (or squared distances or similarity) between
        covariance matrices, to use for the loss. The function should take two batches of
        covariance matrices of shape (batch_size, n_dim, n_dim) and return a tensor of shape
        (batch_size, batch_size).
    epochs : int, optional
        Number of training epochs. By default 30.
    learning_rate : float, optional
        Initial learning rate, by default 0.1.
    decay_step : int, optional
        Number of steps to decay the learning rate, by default 1000.
    decay_rate : float, optional
        Learning rate decay factor, by default 1.

    Returns
    -------
    torch.Tensor
        Tensor containing the loss at each epoch (shape: epochs).
    torch.Tensor
        Tensor containing the training time at each epoch (shape: epochs).
    """
    # Create optimizer and scheduler
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_step, gamma=decay_rate
    )
    optimizer.zero_grad()

    n_classes = model.input_covariances.shape[0]
    tril_ind = torch.tril_indices(n_classes, n_classes, offset=-1)
    loss = []
    training_time = []
    prev_loss = 0.0
    loss_change = 0.0
    total_start_time = time.time()

    for e in tqdm(range(epochs), desc="Epochs", unit="epoch"):

        covariances = model.get_feature_covariances()
        distances = distance_fun(covariances, covariances)
        epoch_loss = -torch.mean(distances)

        epoch_loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_time = time.time() - total_start_time

        # Update tqdm bar description with loss change and total time
        tqdm.write(
            f"Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}, "
            + f"Change: {loss_change:.4f}, Time: {epoch_time:.2f}s"
        )

        training_time.append(epoch_time)

        loss_change = prev_loss - epoch_loss
        prev_loss = epoch_loss.item()

        loss.append(epoch_loss)

    return torch.as_tensor(loss), torch.as_tensor(training_time)
