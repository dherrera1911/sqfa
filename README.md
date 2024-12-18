# SQFA

[![Build Status](https://github.com/dherrera1911/sqfa/actions/workflows/install.yml/badge.svg)](https://github.com/dherrera1911/sqfa/actions/workflows/install.yml)
[![Tests Status](https://github.com/dherrera1911/sqfa/actions/workflows/tests.yml/badge.svg)](https://github.com/dherrera1911/sqfa/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/sqfa/badge/?version=latest)](https://sqfa.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/dherrera1911/sqfa?tab=MIT-1-ov-file)
![Python version](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12-blue.svg)
[![codecov](https://codecov.io/gh/dherrera1911/sqfa/graph/badge.svg?token=NN44R5G18I)](https://codecov.io/gh/dherrera1911/sqfa)
[![PyPI version](https://badge.fury.io/py/sqfa.svg)](https://badge.fury.io/py/sqfa)


Supervised Quadratic Feature Analysis (SQFA) is a supervised dimensionality
reduction technique. It learns a set of linear features that
maximize the differences in second-order statistics between
classes. The `sqfa` package implements SQFA.

For detailed information on the method and the package, see the
[sqfa package tutorials](https://sqfa.readthedocs.io/en/latest/tutorials/spd_geometry.html).

## Overview

The package provides the class `SQFA` that can be used to train the
model. The class has an API similar to `sklearn`.
An example of how to use the `SQFA` class is shown below:

```python
import sqfa
import torchvision

### DOWNLOAD DATASET

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True
)
X = trainset.data.reshape(-1, 28 * 28).float()
y = trainset.targets
# Normalize X
X = X / (X.std() * 28.0)
X = X - X.mean(dim=0, keepdim=True)

### INITIALIZE SQFA MODEL

model = sqfa.model.SQFA(
    n_dim=X.shape[-1],
    n_filters=4,
    feature_noise=0.01,
)

### FIT MODEL. TWO OPTIONS:

# 1) Give data and labels as input
model.fit_pca(X) # Optional: PCA initialization to speed up convergence
model.fit(X=X, y=y)

# 2) Give scatter matrices as input
data_stats = sqfa.statistics.class_statistics(X, y)
second_moments = data_stats["second_moments"]
model.fit_pca(data_scatters=second_moments) # Optional: PCA initialization to speed up convergence
model.fit(data_scatters=second_moments)

### TRANSFORM DATA AND SECOND MOMENTS TO THE LEARNED FEATURE SPACE

X_transformed = model.transform(X).detach()
second_moments_transformed = model.transform_scatters(
    second_moments
).detach()
```

## Installation

### Virtual environment

We recommend installing the package in a virtual environment. For this,
you can first install `miniconda` 
([install instructions link](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)),
and then create a virtual environment with Python 3.11 using the following
shell command:

```bash
conda create -n my-sqfa python=3.11
```

You can then activate the virtual environment with the following command:

```bash
conda activate my-sqfa
```

You should activate the `my-sqfa` environment to install the package, and every
time you want to use it.

### Install package from PyPI

The easiest way to install the package is form the PyPI
repository. To install the package and the dependencies
needed to run the tutorials, use the following command:

```bash
pip install sqfa[dev]
```

To install the lighter version without the tutorials dependencies, use

```bash
pip install sqfa
```

### Install package from source

To install the package from source (e.g., if you want to modify the
code), you can clone the repository and install the package
in editable mode with the following commands:

```bash
git clone git@github.com:dherrera1911/sqfa.git
cd sqfa
pip install -e .[dev]
```
