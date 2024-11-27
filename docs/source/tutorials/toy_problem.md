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

# Feature selection: SQFA vs PCA and LDA

In this tutorial we consider a toy problem to compare SQFA to other standard
feature learning techniques, Principal Component Analysis (PCA) and Linear
DiscrimiAnalysis (LDA).

## SQFA vs PCA: The 0-mean case

Our toy problem has 4 dimensional data and 3 classes.
The data statistics are designed to accentuate the differences
between SQFA, PCA and LDA features. For comparing SQFA and PCA,
the statistics of the toy problem data are as follows:

1) Dimensions 1 and 2 have a covariance structure that is rotated for the
different classes. SQFA will favor these dimensions because
of the second-order differences between classes.
2) Dimensions 3 and 4 have high variance but the same covariance
structure for all classes. PCA will favor these dimensions because
of the high variance.
3) Dimensions 1 and 2 are uncorrelated with dimensions 3 and 4.
4) The means are 0 for all dimensions and classes.

Let's generate the covariances matrices described above and visualize
the data distributions:

```{code-cell} ipython3
import torch
import sqfa
import matplotlib.pyplot as plt

torch.manual_seed(9) # Set seed for reproducibility

# GENERATE COVARIANCES

# Define the functions to generate the covariance matrices
def make_rotation_matrix(theta):
    """Make a matrix that rotates the first 2 dimensions of a 4D tensor"""
    theta = torch.deg2rad(theta)
    rotation = torch.eye(4)
    rotation[:2, :2] = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                     [torch.sin(theta), torch.cos(theta)]])
    return rotation

def make_rotated_classes(base_cov, angles):
    """Rotate the first 2 dimensions of base_cov by the angles in the angles list"""
    covs = torch.as_tensor([])
    for theta in angles:
        rotation_matrix = make_rotation_matrix(theta)
        rotated_cov = torch.einsum('ij,jk,kl->il', rotation_matrix, base_cov, rotation_matrix.T)
        covs = torch.cat([covs, rotated_cov.unsqueeze(0)], dim=0)
    return covs

# Generate the covariance matrices
variances = torch.tensor([0.8, 0.04, 1.0, 1.0])
base_cov = torch.diag(variances)

angles = torch.as_tensor([15, 45, 70])
class_covariances = make_rotated_classes(base_cov, angles)

# VISUALIZE

def plot_data_covariances(ax, covariances, means=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])

    dim_pairs = [[0, 1], [2, 3]]
    legend_type = ['none', 'discrete']
    for i in range(2):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax[i])
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(3),
                               dim_pair=dim_pairs[i], ax=ax[i])
        dim_pairs_label = [d+1 for d in dim_pairs[i]]
        #ax[i].set_title(f'Data space \n dim {dim_pairs_label}', fontsize=12)
        ax[i].set_title(f'Data space (dim {dim_pairs_label})', fontsize=12)
        ax[i].set_aspect('equal')

figsize = (6, 3)
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances)
plt.tight_layout()
plt.show()
```

The left panel shows how the classes have different statistics in
dimensions 1 and 2, and the right panel shows how all classes have
the same covariance in dimensions 3 and 4. The means are all 0.

Let's learn 2 filters each with SQFA and PCA for these class
distributions.[^1]

```{code-cell} ipython
# Learn SQFA filters
model = sqfa.model.SQFA(n_dim=4, n_filters=2, feature_noise=0.01)
model.fit(data_scatters=class_covariances, show_progress=False)
sqfa_filters = model.filters.detach()

# Learn PCA filters
average_cov = torch.mean(class_covariances, dim=0)
eigval, eigvec = torch.linalg.eigh(average_cov)
pca_filters = eigvec[:, -2:].T
```

Let's visualize the filters learned by SQFA and PCA. We plot the filters
as arrows in the original data space, to see how they relate to the
class distributions.

```{code-cell} ipython
# Function to plot filters
def plot_filters(ax, filters, class_covariances, means=None):
    """Plot the filters as arrows in data space."""
    # Plot the statistics of the filters
    plot_data_covariances(ax, class_covariances, means)

    # Draw the filters of sqfa as arrows on the plot
    colors = ['r', 'b']
    awidth = 0.04
    for f in range(2):
        ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])
        ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])

# Plot SQFA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, sqfa_filters, class_covariances)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='center left')
plt.suptitle('SQFA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()

# Plot PCA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, pca_filters, class_covariances)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='center left')
plt.suptitle('PCA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

SQFA filters (top) put all their weight on the dimensions with differences
in covariances (dimensions 1 and 2). PCA filters (bottom) put all their weight
on the dimensions with higher variance (dimensions 3 and 4). Let's
plot the feature space statistics for both methods.

```{code-cell} ipython
# Get feature statistics
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
pca_covariances = torch.einsum('ij,njk,kl->nil', pca_filters, class_covariances, pca_filters.T)

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, ax=ax)
ax.set_title('SQFA feature-space')
ax.set_xlabel('SQFA feature 1')
ax.set_ylabel('SQFA feature 2')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=pca_covariances, ax=ax)
ax.set_title('PCA feature-space')
ax.set_xlabel('PCA feature 1')
ax.set_ylabel('PCA feature 2')
plt.show()
```

We see that the statistics of the classes in the feature space reflect
the subspaces of the data picked by the two methods. The classes have
different covariances in the SQFA feature space, while they have
have higher overall variance in the PCA feature space.


## SQFA vs LDA with different means

LDA is a standard technique for supervised feature learning. It finds the features
that maximize the separation between classes (i.e. between class means) while
minimizing the variability within classes. Next, we compare SQFA with LDA. For
this, we modify the toy problem introducing differences in the means of the classes
(points 1-3 in the starting list remain the same, but point 4 is modified).

```{code-cell} ipython
# Do example with difference in means
class_means = torch.tensor([[0, 0, 1, -1],
                            [0, 0, 0, 1],
                            [0, 0, -1, -1]])
class_means = class_means * 0.4

# Plot the new distributions
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plt.tight_layout()
plt.show()
```

In the new toy problem, the means of the classes are different in the last two
dimensions. Let's obtain the SQFA and LDA filters for this new toy problem.[^2]

```{code-cell} ipython
def lda(scatter_between, scatter_within):
    """Compute LDA filters from between class and within class scatter matrices."""
    eigvec, eigval = sqfa.linalg.generalized_eigenvectors(
      scatter_between,
      scatter_within
    )
    eigvec = eigvec[:, eigval>1e-5]
    return eigvec.transpose(-1, -2)

# Get scatter matrices for LDA
scatter_within = torch.mean(class_covariances, dim=0)
scatter_between = class_means.T @ class_means

# Get second moment matrices for SQFA
mean_outer_prod = torch.einsum('ij,ik->ijk', class_means, class_means)
second_moments = class_covariances + mean_outer_prod

# Learn SQFA
model = sqfa.model.SQFA(n_dim=4, feature_noise=0.01, n_filters=2)
model.fit(data_scatters=second_moments, show_progress=False)
sqfa_filters = model.filters.detach()

# Learn LDA
lda_filters = lda(scatter_between, scatter_within)
```

We next plot the filters learned by SQFA and LDA as arrows in the data
space, like in the previous example:

```{code-cell} ipython
# Plot SQFA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, sqfa_filters, class_covariances, class_means)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('SQFA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()

# Plot LDA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, lda_filters, class_covariances, class_means)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('LDA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

:::{admonition} Loss is invariant to filter sign
The reader may note that the SQFA filters are different in
the first and second example, i.e. Filter 1 changed
sign. The loss function of SQFA is invariant to the sign and
the order of the filters, which we would expect since these
should not affect the second moment differences between classes.
:::

We see that while SQFA filters have their weight on the
dimensions with differences in the covariances (dimensions 1,2)
LDA filters have all their weight on the dimensions with differences
in class means (dimensions 3,4). Let's plot the
statistics of classes in the SQFA and LDA feature space.

```{code-cell} ipython
# Get the means and covariances for the new features
lda_covariances = torch.einsum('ij,njk,kl->nil', lda_filters, class_covariances, lda_filters.T)
lda_means = class_means @ lda_filters.T
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
sqfa_means = class_means @ sqfa_filters.T

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, centers=sqfa_means, ax=ax)
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax)
ax.set_title('SQFA feature-space')
ax.set_xlabel('SQFA feature 1')
ax.set_ylabel('SQFA feature 2')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=lda_covariances, centers=lda_means, ax=ax)
sqfa.plot.scatter_data(data=lda_means, labels=torch.arange(3), ax=ax)
ax.set_title('LDA feature-space')
ax.set_xlabel('LDA feature 1')
ax.set_ylabel('LDA feature 2')
plt.tight_layout()
plt.show()
```

The distribution of SQFA features have different covariances across classes,
while the distribution of LDA features have different means. This
reflects the weights of the filters seen above. Which features are better
for classification will depend on the specifics of
the class distributions and on the classifier used.


## SQFA is sensitive to covariances and means

In the previous example SQFA prioritized the differences in
covariances over the differences in means. However, this is not always the case.
Particularly, note that we fitted SQFA using the second moment matrices of the
classes, which for a given class $i$ are given
by $\Psi_i = \Sigma_i + \mu_i \mu_i^T$
(see [this note](#centered-vs-non-centered) about centered vs non-centered second moments).
Thus, the second moments of a class will
be influenced by both the covariance matrix and the mean of the class.

We can see this by modifying the toy example above to have larger differences
in the means between the first two dimensions. Let's make the class means
more different and plot the new distributions.

```{code-cell} ipython
# Make example with more different means
class_means = torch.tensor([[0, 0, 1, -1],
                            [0, 0, 0, 1],
                            [0, 0, -1, -1]])
class_means = class_means * 2.5

# Plot the new distributions
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plt.tight_layout()
plt.show()
```

Let's learn the SQFA and LDA filters again and visualize them:

```{code-cell} ipython
# Get the new second moment matrices
mean_outer_prod = torch.einsum('ij,ik->ijk', class_means, class_means)
second_moments = class_covariances + mean_outer_prod

# Learn SQFA
model = sqfa.model.SQFA(n_dim=4, feature_noise=0.01, n_filters=2)
model.fit(data_scatters=second_moments, show_progress=False)
sqfa_filters = model.filters.detach()

# Learn LDA
lda_filters = lda(scatter_between, scatter_within)

# Plot SQFA filters
fm = 3 # filter magnification for visualization
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, sqfa_filters * fm, class_covariances, class_means)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('SQFA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()

# Plot LDA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True) 
plot_filters(ax, lda_filters * fm, class_covariances, class_means)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('LDA filters', fontsize=16, x=0.42)
plt.tight_layout()
plt.show()
```

Let's plot again the statistics in the feature space for SQFA and LDA:

```{code-cell} ipython
# Get the means and covariances for the new features
lda_covariances = torch.einsum('ij,njk,kl->nil', lda_filters, class_covariances, lda_filters.T)
lda_means = class_means @ lda_filters.T
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
sqfa_means = class_means @ sqfa_filters.T

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, centers=sqfa_means, ax=ax)
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax)
ax.set_title('SQFA feature-space')
ax.set_xlabel('SQFA feature 1')
ax.set_ylabel('SQFA feature 2')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sqfa.plot.statistics_ellipses(ellipses=lda_covariances, centers=lda_means, ax=ax)
sqfa.plot.scatter_data(data=lda_means, labels=torch.arange(3), ax=ax)
ax.set_title('LDA feature-space')
ax.set_xlabel('LDA feature 1')
ax.set_ylabel('LDA feature 2')
plt.tight_layout()
plt.show()
```
  
This example illustrates that, when using the non-centered second
moment matrices, SQFA is sensitive to both
the covariances and the means of the classes, and that the features learned
will depend on the specifics of the class distributions.

## Conclusion

SQFA learns features that maximize the differences in the second moments 
between classes. These features are different than those learned by other standard techniques like PCA and LDA. SQFA is particularly useful when the
differences in covariances between classes are important for classification.

[^1]: PCA operates on the global covariance matrix (i.e. for the full dataset).
Here we obtain the global covariance matrix by averaging across the
covariance matrices of the different classes. We then compute the eigenvectors
of this average covariance matrix to obtain the PCA filters.

[^2]: LDA filters can be obtained by solving the generalized eigenvalue problem
between the scatter matrix for the class means and the within-class covariance.
For more information see 
[here](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Multiclass_LDA).
