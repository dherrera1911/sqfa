"""Routine to fit SQFA filters using Gradient Descent."""

import time

import torch
from torch import optim
from tqdm import tqdm

__all__ = ["fitting_loop"]


def __dir__():
    return __all__


def fitting_loop(
    model,
    distance_fun,
    epochs=50,
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
    distance_fun : callable
        Function returning pairwise distances between covariance matrices.
        Takes as input two batches of covariance matrices of shape (batch_size, n_dim, n_dim)
        and return a tensor of shape (batch_size, batch_size).
    epochs : int, optional
        Number of training epochs. By default 30.
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

    n_classes = model.input_covariances.shape[0]
    tril_ind = torch.tril_indices(n_classes, n_classes, offset=-1)
    loss_list = []
    training_time = []
    total_start_time = time.time()

    prev_loss = 0.0
    loss_change = -1.0 # Arbitrary value to enter the loop

    def closure():
        optimizer.zero_grad()
        covariances = model.get_feature_covariances()
        distances = distance_fun(covariances, covariances)
        epoch_loss = -torch.mean(distances[tril_ind[0], tril_ind[1]])
        epoch_loss.backward()
        return epoch_loss

    for e in tqdm(range(epochs), desc="Epochs", unit="epoch", disable=not show_progress):

        epoch_loss = optimizer.step(closure)
        epoch_time = time.time() - total_start_time
        loss_change = prev_loss - epoch_loss.item()

        # Update tqdm bar description with loss change and total time
        #tqdm.write(
        #    f"Epoch {e+1}/{epochs}, Loss: {epoch_loss.item():.4f}, "
        #    f"Change: {loss_change:.4f}, Time: {epoch_time:.2f}s"
        #)

        # Break if loss change is below atol
        if loss_change <= atol and loss_change >= 0:
            tqdm.write(
              f"Loss change below {atol}, stopping training at epoch {e+1}/{epochs}."
            )
            break

        prev_loss = epoch_loss.item()
        training_time.append(epoch_time)
        loss_list.append(epoch_loss.item())

    if return_loss:
        return torch.tensor(loss_list), torch.tensor(training_time)
    else:
        return None
