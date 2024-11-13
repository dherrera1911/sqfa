# SQFA

The `sqfa` package implements the Supervised Quadratic Feature Analysis
(SQFA), a dimensionality reduction technique to learn a linear
transformation that maximizes the between-class
differences in second-order statistics (i.e. the second moments of the data).

The uses a geometric approach, where the (centered or uncentered)
second moment matrix of the transformed data for each class is considered as
a point in the manifold of symmetric positive definite (SPD) matrices. SQFA
finds the linear transformation that maximizes the distance between the
second moment matrices of different classes in the manifold. This
method is closely related to geometry-aware PCA.

## Overview

The `sqfa` package provides a class `SQFA` that can be used to
train the SQFA model. The class has an API similar to that of
`sklearn` models, with `fit` and `transform` methods.

An example of how to use the `SQFA` class is shown below:

```python
import sqfa

# Download data

# Compute class-specific second moments

# Initialize SQFA model
model = sqfa.model.SQFA(
    input_covariances=covariances,
    feature_noise=0.01,
    n_filters=4,
)

# Fit model
model.fit()

# Transform data
transformed_data = model.transform(data)
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

