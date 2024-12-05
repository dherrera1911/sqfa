---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Distances in SPD manifold

As mentioned in the [geometry tutorial](https://sqfa.readthedocs.io/en/latest/tutorials/spd_geometry.html),
the manifold of Symmetric Positive Definite (SPD) matrices,
$\mathcal{S}^m_{++}$, is compatible with different Riemannian metrics
(see [Congedo and Jain 2019](https://hal.science/hal-02315153/document)
for a concise overview of the most common metrics). In the SQFA paper
the Affine Invariant Riemannian Metric (AIRM) is linked to
class discriminability, and this is the default metric in the
`sqfa` package. However, it may be interesting to explore other
metrics, either for theoretical reasons or for computational
efficiency. In this tutorial we test some relevant metrics,
and show how you can implement your own metric to use with
the `sqfa` package.

:::{admonition} Riemannian metrics vs distances
We use the terms "metric" and "distance" somewhat interchangeably
in this tutorial, but they are not the same.

In simplified terms, we can think of the metric as telling us
how to measure **speeds** of curves on the manifold.
Like in Euclidean space, the length of a curve is obtained by
integrating the speed along the curve. The distance between two
points in a Riemannian manifold is the length of the shortest
curve connecting them.

The metric is the more fundamental concept in differential geometry,
and a given metric defines a distance function on the manifold. Because the
metric is more fundamental, we often use the term "metric" to refer
to the geometry that we are using, although what we really care about in SQFA
is the distance function.
:::


## Affine Invariant Riemannian Metric (AIRM)

For two SPD matrices $A$ and $B$, the Affine Invariant
distance is defined as:

$$d_{\text{AIRM}}(A, B) = \left( \sum_{i=1}^m \log^2(\lambda_i) \right)^{1/2} =
\| \log(A^{-1/2} B A^{-1/2}) \|_F$$

where in the first definition $\lambda_i$ are the
[generalized eigenvalues](https://arxiv.org/pdf/1903.11240)
of the pair $(A,B)$, and in the second definition $\|\|_F$ is the
Frobenius norm, $\log$ is the matrix logarithm, and $A^{-1/2}$ is the
matrix square root of $A$.

In the SQFA paper we establish theoretical links between the
Affine Invariant distance and second-order class discriminability
([Herrera-Esposito and Burge, NeurReps Workshop 2024](https://openreview.net/pdf?id=vxdPzM18Xx)).
Thus, the Affine Invariant distance is the default distance in the
`sqfa` package. 

:::{admonition} AIRM metric and Fisher-Rao metric for Gaussian distributions
The Fisher-Rao metric is a Riemannian metric in manifolds of probability
distributions (i.e. where each point is a probability distribution).
Under the Fisher-Rao metric, the squared "speed" of a curve at a given
point $\theta$ (we can think of $\theta$ as the parameters of a probability
distribution) moving with velocity vector $\Psi$ is given by the
Fisher information of $\theta$ along the direction $\Psi$.

Leaving aside the technical details, the above means that the
Fisher-Rao metric tells us that speeds are large when the change to
the probability distribution is easy to detect, and small when the change
is hard to detect.

Interestingly, the AIRM metric for SPD matrices is equivalent to the
Fisher-Rao metric for zero-mean Gaussian distributions. Thus, the AIRM
distance applied to second-moment matrices has some intepretability in
terms of probability distributions: it tells us how discriminable
are the infinitesimal changes transforming the zero-mean Gaussian
given by $A$ into the zero-mean Gaussian given by $B$.
:::


The Affine Invariant distance is invariant to affine transformations,
meaning that if $C$ is an invertible matrix, then
$d_{\text{AIRM}}(A, B) = d_{\text{AIRM}}(C A C^T, C B C^T)$.
This means that the distance is invariant to transformations
such as change of basis and scalings. The distance is also invariant
to inversions, i.e., $d_{\text{AIRM}}(A, B) = d_{\text{AIRM}}(A^{-1}, B^{-1})$.
These are desirable properties for many applications.

The squared AIRM distance is implemented in
`sqfa.distances.affine_invariant_sq()`. Let's create some SPD
matrices to test this function. Like in the
[Feature selection](https://sqfa.readthedocs.io/en/latest/tutorials/toy_problem.html)
tutorial, we create covariances for different classes by having a base
covariance and rotating along different dimensions. We make a set of 3
SPD matrices of size $4 \times 4$ with the following properties:

1) The ellipses of the different classes in dimensions (1,2) have low variance,
and large rotations
2) The ellipses of the different classes in dimensions (3,4) have high variance,
and low rotation angles
3) The ellipses in dimensions (1,2) and (3,4) have the same aspect ratio

```{code-cell} ipython3
import torch
import sqfa
import matplotlib.pyplot as plt

torch.manual_seed(9) # Set seed for reproducibility
n_dim_pairs = 2

# GENERATE COVARIANCE MATRICES
# Define the functions to generate the covariance matrices
def make_rotation_matrix(theta, dims):
    """Make a matrix that rotates 2 dimensions of a 6x6 matrix by theta.
    
    Args:
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    theta = torch.deg2rad(theta)
    rotation = torch.eye(n_dim_pairs*2)
    rot_mat_2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                              [torch.sin(theta), torch.cos(theta)]])
    for row in range(2):
        for col in range(2):
            rotation[dims[row], dims[col]] = rot_mat_2[row, col]
    return rotation

def make_rotated_classes(base_cov, angles, dims):
    """Rotate 2 dimensions of base_cov, specified in dims, by the angles in the angles list
    Args:
        base_cov (torch.Tensor): Base covariances
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    if len(angles) != base_cov.shape[0]:
        raise ValueError('The number of angles must be equal to the number of classes.')

    for i, theta in enumerate(angles):
        rotation_matrix = make_rotation_matrix(theta, dims)
        base_cov[i] = torch.einsum('ij,jk,kl->il', rotation_matrix, base_cov[i], rotation_matrix.T)
    return base_cov

angles = [
  [0, 40, 80], # Dimensions 1, 2
  [0, 20, 40],  # Dimensions 3, 4
]

n_angles = len(angles[0])
variances = torch.tensor([0.25, 0.01, 1.0, 0.04])
base_cov = torch.diag(variances)
base_cov = base_cov.repeat(n_angles, 1, 1)

class_covariances = base_cov
for d in range(len(angles)):
    ang = torch.tensor(angles[d])
    class_covariances = make_rotated_classes(
      class_covariances, ang, dims=[2*d, 2*d+1]
    )

# VISUALIZE
def plot_data_covariances(ax, covariances, means=None, lims=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])
    n_classes = means.shape[0]

    dim_pairs = [[0, 1], [2, 3]]
    legend_type = ['none', 'discrete']
    for i in range(len(dim_pairs)):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax[i])
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(n_classes),
                               dim_pair=dim_pairs[i], ax=ax[i])
        dim_pairs_label = [d+1 for d in dim_pairs[i]]
        ax[i].set_title(f'Data space (dim {dim_pairs_label})', fontsize=12)
        ax[i].set_aspect('equal')
        if lims is not None:
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)

figsize = (8, 4)
lims = (-2.2, 2.2)
fig, ax = plt.subplots(1, n_dim_pairs, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, lims=lims)
plt.tight_layout()
plt.show()
```

We see that with these covariance matrices, dimensions (1,2) have
lower variance but the classes are better separated (we'll come
back to this point later).

The tensor `class_covariances` with the covariance matrices of the
different classes has shape `(3, 4, 4)`. The first dimension is
a batch dimensions, and the second and third dimensions are the
dimensions of the covariance matrices. We next use the function
`sqfa.distances.affine_invariant_sq()` to compute the squared AIRM
distance between each pair of matrices, and print the
distance matrix

```{code-cell} ipython3
# COMPUTE DISTANCES BETWEEN COVARIANCES
ai_dist_sq = sqfa.distances.affine_invariant_sq(
  A=class_covariances, B=class_covariances
)

# PRINT DISTANCE MATRIX
torch.set_printoptions(precision=1)
print(f"Matrix of AIRM squared distances in manifold:\n{ai_dist_sq.round(decimals=2)}")
```

We can see that the inputs to `sqfa.distances.affine_invariant_sq()`
were two tensors with the same shape `(3, 4, 4)`. The function
returns a tensor with shape `(3, 3)`, that is `(batch_A, batch_B)`,
with all the pairwise squared distances. 

As expected, the diagonal elements of the output matrix are zero,
because that's the distance of a matrix to itself. We also see that,
as expected, the matrix is symmetric, because the distance is symmetric.

:::{admonition} Implementing custom distance functions
It is important to note the input-output signature of the
`sqfa.distances.affine_invariant_sq()` function discussed in the text.
The only requirement to use a custom distance function with
`sqfa` is that it has the same input-output signature. That is,
the distance function should take as input two tensors
with batch dimensions `batch_A` and `batch_B`, and return a tensor
of pairwise distances with shape `(batch_A, batch_B)`.
:::


Let's use SQFA to learn the 2 filters that maximize the AIRM
squared distance between the classes. Because of the link between the AIRM
distance and class discriminability, we expect that the features
will span the first two dimensions, which have the least overlap
between classes and are thus the most discriminative.


```{code-cell} ipython
# LEARN FILTERS WITH AIRM DISTANCE
noise = 0.001
n_dim = class_covariances.shape[-1]
sqfa_airm = sqfa.model.SQFA(
  n_dim=n_dim, n_filters=2, feature_noise=noise,
  distance_fun = sqfa.distances.affine_invariant_sq,
)
sqfa_airm.fit(data_scatters=class_covariances, show_progress=False)
airm_filters = sqfa_airm.filters.detach()
```

Let's visualize the filters as arrows pointing in the data space:


```{code-cell} ipython
# VISUALIZE FILTERS ON TOP OF DATA COVARIANCES
def plot_filters(ax, filters, class_covariances, means=None):
    """Plot the filters as arrows in data space."""
    # Plot the statistics of the filters
    plot_data_covariances(ax, class_covariances, means, lims=lims)

    # Draw the filters of sqfa as arrows on the plot
    colors = ['r', 'b']
    awidth = 0.04
    for f in range(2):
        ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])
        ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])

# Plot AIRM filters
fig, ax = plt.subplots(1, n_dim_pairs, figsize=figsize, sharex=True, sharey=True)
plot_filters(ax, airm_filters, class_covariances)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='center left')
plt.suptitle('AIRM filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

As expected, the filters span the dimensions (1,2), and thus they are
the features with highest second-order class discriminability.

## Bures-Wasserstein Metric: Implementing a custom distance function

Another important metric commonly used in the SPD manifold is the
Bures-Wasserstein (BW) metric, defined as:

$$d_{\text{BW}}(A, B) = \sqrt{\text{tr}(A) + \text{tr}(B) -
2\text{tr}\left(\sqrt{A^{1/2} B A^{1/2}}\right)}$$

where $\text{tr}$ is the trace operator. Like the AIRM distance, the
BW metric is related to the geometry of 0-mean Gaussian
distributions as mentioned in the note below.

:::{admonition} BW metric and optimal transport of Gaussian distributions
:name: optimal-transport
Like the AIRM distance, the BW metric in the SPD manifold has an interpretation
in terms of Gaussian distributions. Specifically, the BW distance between two
SPD matrices $A$ and $B$ is the optimal transport distance between the two
zero-mean Gaussian distributions with covariance matrices $A$ and $B$.

The optimal transport distance is also known as the earth mover's
distance, and it can be thought of as the cost of moving the mass
from one distribution to the other. I.e. imagine that the
Gaussian distribution given by covariance matrix $A$ is a pile of
dirt, then the BW distance is the cost of moving the dirt into the
shape given by covariance matrix $B$. Optimal transport distances
are a popular tool in machine learning, and sometimes have
advantages with respect to the Fisher-Rao metric.

From the earth mover's perspective, it is easy to see that the BW
distance is not scale-invariant: if we scale up the distributions,
we need to move the dirt across larger distances, increasing the cost.
:::

Let's implement a function that computes the BW squared distance[^1] to
use for learning features with SQFA. 

```{code-cell} ipython
# IMPLEMENT BURES WASSERSTEIN SQUARED DISTANCE
def bw_sq(A, B):
    """Compute the Bures-Wasserstein squared distance between all pairs
    of matrices in A and B."""
    tr_A = torch.einsum('ijj->i', A)
    tr_B = torch.einsum('ijj->i', B)
    A_sqrt = sqfa.linalg.spd_sqrt(A)
    C = sqfa.linalg.conjugate_matrix(B, A_sqrt)
    C_sqrt_eigvals = torch.sqrt(torch.linalg.eigvalsh(C))
    tr_C = torch.sum(C_sqrt_eigvals, dim=-1)
    bw_distance_sq = tr_A[None,:] + tr_B[:,None] - 2 * tr_C
    return torch.abs(bw_distance_sq)

# COMPUTE BW DISTANCES
bw_dist_sq = bw_sq(class_covariances, class_covariances)
print(f"Matrix of BW squared distances in manifold:\n{bw_dist_sq.round(decimals=2)}")
```

We see that like the AIRM distance function, this function takes two
tensors with batch dimensions `batch_A` and `batch_B`, and returns
a tensor with shape `(batch_A, batch_B)`. The values in the diagonal
that have the self-distances are 0, and the matrix is symmetric
as expected. Let's now use the BW distance to learn features with SQFA,
by passing this new function to the `distance_fun` argument of the
`sqfa.model.SQFA` class.

```{code-cell} ipython
# LEARN FILTERS WITH BURES WASSERSTEIN DISTANCE
sqfa_bw = sqfa.model.SQFA(
  n_dim=n_dim, n_filters=2,
  feature_noise=noise,
  distance_fun = bw_sq,
)
sqfa_bw.fit(data_scatters=class_covariances, show_progress=False)

bw_filters = sqfa_bw.filters.detach()

# Plot BW metric filters
fig, ax = plt.subplots(1, n_dim_pairs, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, bw_filters, class_covariances)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='center left')
plt.suptitle('BW filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

The BW filters put their weights in dimensions (3,4) which are the less
discriminable pair, unlike the AIRM filters. Why is this the case? Using the
[earth mover's intuition](#optimal-transport) of the BW distance, we
can see that the BW distance is not scale-invariant. In our toy problem,
the dimensions (3,4) have higher variance, and thus the cost of moving
the dirt from one distribution to the other is higher. This gives us an
intuition of why the BW distance might not prioritize the most discriminable
features. There might be reasons to want filters that maximize the BW distance,
but the AIRM metric seems more aligned with the goal of maximizing class
discriminability.


## Euclidean distance

Finally, let's see how the most naive distance function, the Euclidean
distance, performs in this problem. The Euclidean distance between two
SPD matrices $A$ and $B$ is defined as the Frobenius norm of the difference
between the matrices:

$$d_{\text{Euclidean}}(A, B) = \| A - B \|_F = \sqrt{\sum_{i,j} (A-B)_{i,j}^2}$$

Let's implement this distance function and learn features with SQFA using
this distance.

```{code-cell} ipython
def euclidean_sq(A, B):
    """Compute the Euclidean distance between all pairs of matrices in A and B."""
    diff_sq = (A.unsqueeze(0) - B.unsqueeze(1))**2
    euclidean_distance_sq = torch.sum(diff_sq, dim=(-1,-2))
    return euclidean_distance_sq

# LEARN FILTERS WITH EUCLIDEAN SQUARED DISTANCE
sqfa_eq = sqfa.model.SQFA(
  n_dim=n_dim, n_filters=2,
  feature_noise=noise,
  distance_fun = euclidean_sq,
)
sqfa_eq.fit(data_scatters=class_covariances, show_progress=False)

eq_filters = sqfa_eq.filters.detach()

# Plot Euclidean metric filters
fig, ax = plt.subplots(1, n_dim_pairs, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, eq_filters, class_covariances)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='center left')
plt.suptitle('Euclidean filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

The Euclidean filters again put their weights in the dimensions (3,4),
but not only that, they are also perfectly parallel to each other,
meaning that the filters are completely redundant. This redundancy will
make the feature covariance matrices have the highest possible off-diagonal
elements, which will make for larger Euclidean differences between the
elements, but will also make the classes less discriminable.

Thus, although we are maximizing some distance between second-moment matrices
in the SPD manifold, the Euclidean distance is not a good choice for this
problem. This is a good example of how the choice of distance function
is crucial in the success of the feature learning process.


[^1]: For efficiency in implementing the squared distance we use a couple of linear algebra tricks. First, we use that $\text{tr}(M)$ is the sum of the eigenvalues of $M$. Then, we use that $A^{1/2} B A^{1/2}$ is SPD, because $A^{1/2}$ is symmetric, and any SPD matrix $M$ and invertible matrix $G$ satisfy that $G M G^T$ is SPD. Finally, we use that for an SPD matrix $M$, the eigenvalues of $\sqrt{M}$ are the square roots of the eigenvalues of $M$. Thus, we have that $\text{tr}(A^{1/2} B A^{1/2}) = \sum_i \lambda_i^{1/2}$, where $\lambda_i$ are the eigenvalues of $A^{1/2} B A^{1/2}$.

