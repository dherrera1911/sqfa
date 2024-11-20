# SQFA

Supervised Quadratic Feature Analysis (SQFA) is a linear supervised dimensionality
reduction technique maximizing the differences in second-order statistics between
classes. The `sqfa` package provides an implementation of the SQFA algorithm in PyTorch.

SQFA uses a geometric loss on the class-specific second moment matrices,
considered as points in the SPD manifold. Intuitively, the distance between
the SPD matrices of different classes is a measure of their second-order
dissimilarity. SQFA finds the features that maximize this distance.

## Overview

The `sqfa` package provides a class `SQFA` that can be used to train the
model. The class has an API similar to that of `sklearn` models.
An example of how to use the `SQFA` class is shown below:

```python
import sqfa
import torchvision

### Download dataset

trainset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True
)
x = trainset.data.reshape(-1, 28 * 28).float()
y = trainset.targets
# Normalize x
x = x / torch.linalg.norm(x, dim=1, keepdim=True)

### Initialize SQFA model

model = sqfa.model.SQFA(
    n_dim=28*28,
    n_filters=4,
    feature_noise=0.001,
)

### Fit model. Two options:

# 1) Give data and labels as input
model.fit(X=x, y=y)

# 2) Give scatter matrices as input
data_stats = sqfa.linalg_utils.class_statistics(x, y)
model.fit(data_scatters=data_stats["second_moments"])

### Transform data to the learned feature space

x_transformed = model.transform(x).detach()
```

See the tutorials for more details on the model usage and behavior.

## Installation

To install the package, clone the repository, go to the
downloaded directory and install using pip. In the command
line, this can be done as follows:

```bash
git clone git@github.com:dherrera1911/sqfa.git
cd sqfa
pip install -e .
```

We recommend installing the package in a virtual
environment (e.g. using `conda`). For more detailed instructions, see the
installation section of the tutorials.

