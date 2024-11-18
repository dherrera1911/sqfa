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

# SQFA vs PCA and LDA - Toy Problem

In this tutorial we build a simple toy problem to illustrate the types of features
that are learnt by SQFA, and how they compare to other standard feature
learning techniques.

## SQFA vs PCA in the 0-mean case

To compare the methods, we will generate a toy problem with 3 classes over a 4
dimensional space. We will introduce different statistical properties across
the first two and the last two dimensions of the data space, and we will test
which dimensions are emphasized by SQFA vs PCA and Linear Discriminant Analysis (LDA).

SQFA finds the features that maximize the difference in second-order statistics
between the classes. PCA, on the other hand, finds the features that
maximize the variance of the data. In our first example, we will generate
statistics for 3 classes with the following properties:

- Dimensions 1 and 2 have high variance, but the same covariance
  structure for all classes.
- Dimensions 3 and 4 have lower variance, but the covariances are different
  for each class.
- Dimensions 1 and 2 are uncorrelated with dimensions 3 and 4.
- The means are 0 for all dimensions and classes.

We implement the statistics above by having an initial covariance matrix, and
progressively rotating the covariance of dimensions 3 and 4 for each
class. Let's first define the function that makes the rotated covariance

```{code-cell} ipython3
import torch
import sqfa
import matplotlib.pyplot as plt

def make_rotation_matrix(theta):
    """Make a matrix that rotates the last 2 dimensions of a 4D tensor"""
    theta = torch.deg2rad(theta)
    rotation = torch.eye(4)
    rotation[2:, 2:] = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                     [torch.sin(theta), torch.cos(theta)]])
    return rotation

def make_rotated_classes(base_cov, angles):
    """Rotate the last 2 dimensions of base_cov by the angles in the angles list"""
    covs = torch.as_tensor([])
    for theta in angles:
        rotation_matrix = make_rotation_matrix(theta)
        rotated_cov = torch.einsum('ij,jk,kl->il', rotation_matrix, base_cov, rotation_matrix.T)
        covs = torch.cat([covs, rotated_cov.unsqueeze(0)], dim=0)
    return covs
```

Let's generate the covariance matrices:

```{code-cell} ipython3
# Base diagonal covariance
variances = torch.tensor([1.0, 1.0, 0.8, 0.02])
base_cov = torch.diag(variances)

angles = torch.as_tensor([15, 45, 70])
class_covariances = make_rotated_classes(base_cov, angles)
```

And let's now make a function to visualize our 4D covariances:

```{code-cell} ipython3
def plot_data_covariances(ax, covariances, means=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])

    dim_pairs = [[0, 1], [2, 3]]
    legend_type = ['none', 'discrete']
    for i in range(2):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax[i],
                                      legend_type=legend_type[i])
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(3),
                               dim_pair=dim_pairs[i], ax=ax[i])
        dim_pairs_label = [d+1 for d in dim_pairs[i]]
        ax[i].set_title(f'Data statistics: dim {dim_pairs_label}')
        ax[i].set_aspect('equal')

figsize = (7, 3.5)
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances)
plt.tight_layout()
plt.show()
```

The left panel shows how dimensions 1 and 2 have the same covariance for all
classes, and the right panel shows how dimensions 3 and 4 have different
covariances for each class.

Let's learn 2 features using PCA and SQFA for this problem. To
obtain the PCA features we average the covariance matrices across classes
to obtain the dataset covariance
and [compute its eigenvectors.](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#covariance-matrix)

```{code-cell} ipython
# Learn SQFA filters
model = sqfa.model.SQFA(n_dim=4, n_filters=2, feature_noise=0.001)
model.fit(data_scatters=class_covariances, show_progress=False)
sqfa_filters = model.filters.detach()

# Learn PCA filters
average_cov = torch.mean(class_covariances, dim=0)
eigval, eigvec = torch.linalg.eigh(average_cov)
pca_filters = eigvec[:, -2:].T
```

Let's visualize the filters learned by SQFA and PCA. We plot the filters
as arrows in the original data space, onto which the data is projected.

```{code-cell} ipython
# Function to plot filters
def plot_filters(ax, filters):
    """Plot the filters as arrows in data space."""
    # Draw the filters of sqfa as arrows on the plot
    colors = ['r', 'b']
    awidth = 0.02
    for f in range(2):
        ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])
        ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                    head_width=awidth*5, label=f'Filter {f}', color=colors[f])
    # Add legend outside of the plot, to the right
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

# Plot PCA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances)
plot_filters(ax, pca_filters)
plt.suptitle('PCA filters', fontsize=16)
plt.tight_layout()
plt.show()

# Plot SQFA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances)
plot_filters(ax, sqfa_filters)
plt.suptitle('SQFA filters', fontsize=16)
plt.tight_layout()
plt.show()
```

While PCA filters put all their weight into the dimensions
with higher variance (1 and 2), SQFA filters put all their weight
into the dimensions with differences in covariances (3 and 4).
Let's next visualize the statistics of the features learned by both methods.

```{code-cell} ipython
# Get feature statistics
pca_covariances = torch.einsum('ij,njk,kl->nil', pca_filters, class_covariances, pca_filters.T)
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)

kwargs = {
    'bbox_to_anchor': (1.05, 1),
    'loc': 'upper left',
    'borderaxespad': 0.0,
    'title': 'Class',
} # Legend arguments

fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
sqfa.plot.statistics_ellipses(ellipses=pca_covariances, ax=ax[0])
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, ax=ax[1], legend_type='discrete', **kwargs)
ax[0].set_title('PCA feature statistics')
ax[1].set_title('SQFA feature statistics')
ax[0].set_xlabel('Feature 1')
ax[0].set_xlabel('Feature 2')
ax[1].set_xlabel('Feature 1')
ax[1].set_xlabel('Feature 2')
plt.tight_layout()
plt.show()
```

We see that while PCA features miss the differences in covariances between classes
(left), SQFA features capture these differences (right). With the use of an appropriate
classifier, SQFA features should be able to separate the classes better than PCA.


## SQFA vs LDA with different means

LDA is a standard technique for supervised feature learning. It finds the features
that maximize the variability between classes (i.e. between class means) while
minimizing the variability within classes. Next, we compare SQFA with LDA.

In our previous example there were no differences in the means, so LDA would not be
able to find any useful features. Thus, we generate some class means that differ
in the first two dimensions and visualize the new distributions.

```{code-cell} ipython
# Do example with difference in means
class_means = torch.tensor([[1, -1, 0, 0],
                            [0, 1, 0, 0],
                            [-1, -1, 0, 0]])
class_means = class_means * 0.4

# Plot the new distributions
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plt.tight_layout()
plt.show()
```

Now the means of the classes are different in the first two dimensions.
Let's learn the filters with LDA and SQFA now and compare the results. First, we
define a function to learn the LDA filters using
[generalized eigenvectors](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Multiclass_LDA).

```{code-cell} ipython
def lda(scatter_between, scatter_within):
    """Compute LDA filters from between class and within class scatter matrices."""
    eigvec, eigval = sqfa.linalg_utils.generalized_eigenvectors(
      scatter_between,
      scatter_within
    )
    eigvec = eigvec[:, eigval>1e-5]
    return eigvec.transpose(-1, -2)
```

Now we learn the filters with SQFA and LDA. We note that first we compute the
between and within class scatter matrices for LDA, as well as the class
second moment matrices for SQFA.

```{code-cell} ipytho
# Get scatter matrices for LDA
scatter_within = torch.mean(class_covariances, dim=0)
scatter_between = class_means.T @ class_means

# Get second moment matrices for SQFA
mean_outer_prod = torch.einsum('ij,ik->ijk', class_means, class_means)
second_moments = class_covariances + mean_outer_prod

# Learn LDA
lda_filters = lda(scatter_between, scatter_within)

# Learn SQFA
model = sqfa.model.SQFA(n_dim=4, feature_noise=0.001, n_filters=2)
model.fit(data_scatters=second_moments, show_progress=False)
sqfa_filters = model.filters.detach()
```

Like in the previous example, we plot the filters learned by LDA and SQFA:

```{code-cell} ipython
# Plot LDA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plot_filters(ax, lda_filters)
plt.suptitle('LDA filters', fontsize=16)
plt.tight_layout()
plt.show()

# Plot SQFA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plot_filters(ax, sqfa_filters)
plt.suptitle('SQFA filters', fontsize=16)
plt.tight_layout()
plt.show()
```

As expected, LDA filters have all their weight on the dimensions with differences
in class means (dimensions 1,2), while SQFA filters have their weight on the
dimensions with differences in the covariances (dimensions 3,4). Let's plot the
statistics of the features learned by both methods.

```{code-cell} ipython
# Get the means and covariances for the new features
lda_covariances = torch.einsum('ij,njk,kl->nil', lda_filters, class_covariances, lda_filters.T)
lda_means = class_means @ lda_filters.T
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
sqfa_means = class_means @ sqfa_filters.T

kwargs = {
    'bbox_to_anchor': (1.05, 1),
    'loc': 'upper left',
    'borderaxespad': 0.0,
    'title': 'Class',
} # Legend arguments

fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

# LDA statistics
sqfa.plot.statistics_ellipses(ellipses=lda_covariances, centers=lda_means,
                              ax=ax[0])
sqfa.plot.scatter_data(data=lda_means, labels=torch.arange(3), ax=ax[0])

# SQFA statistics
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, centers=sqfa_means,
                              ax=ax[1], legend_type='discrete', **kwargs)
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax[1])

ax[0].set_title('LDA feature statistics')
ax[1].set_title('SQFA feature statistics')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')
plt.tight_layout()
plt.show()
```

We see that the distribution of LDA features have different means, while
the distribution of SQFA features have the same means but different covariances.
Which features are better for classification will depend on the specifics of
the class distributions and on the classifier used.


## SQFA is sensible to covariances and means

In the previous example we saw that SQFA prioritized the differences in
covariances over the differences in means. However, this is not always the case.
Particularly, note that we fitted SQFA using the second moment matrices of the
classes, which for a given class $i$ are given
by $\Psi_i = \Sigma_i + \mu_i \mu_i^T$. Thus, the second moments of a class will
be influenced by both the covariance matrix and the mean of the class.

We can see this by modifying the toy example above to have larger differences
in the means between the first two dimensions. Let's make the class means
more different and plot the new distributions.

```{code-cell} ipython
# Make example with more different means
class_means = torch.tensor([[1, -1, 0, 0],
                            [0, 1, 0, 0],
                            [-1, -1, 0, 0]])
class_means = class_means * 4.0

# Plot the new distributions
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plt.tight_layout()
plt.show()
```

Let's learn the SQFA filters again and visualize them:

```{code-cell} ipython
# Get the new second moment matrices
mean_outer_prod = torch.einsum('ij,ik->ijk', class_means, class_means)
second_moments = class_covariances + mean_outer_prod

# Learn SQFA
model = sqfa.model.SQFA(n_dim=4, feature_noise=0.001, n_filters=2)
model.fit(data_scatters=second_moments, show_progress=False)
sqfa_filters = model.filters.detach()

# Plot SQFA filters
fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plot_filters(ax, sqfa_filters*3) # Make arrows longer for better visualization
plt.suptitle('SQFA filters', fontsize=16)
plt.tight_layout()
plt.show()
```

The filters learned by SQFA put more weight on the dimensions
with differences in the means now, unlike the previous example.
Let's plot the statistics of the SQFA features for this new example:

```{code-cell} ipython
# Get the means and covariances for the new features
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
sqfa_means = class_means @ sqfa_filters.T

# Plot the new features
fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances,
                              centers=sqfa_means, ax=ax)
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax)
ax.set_title('SQFA features')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```
  
This example illustrates that SQFA is sensitive to both
the covariances and the means of the classes, and that the features learned
will depend on the specifics of the class distributions. Note that this is
the case if we use the second moment matrices as input to SQFA. If we use
only the covariance matrices, SQFA will only be sensitive to the differences
in the covariances between classes.

## Conclusion

SQFA learns features that maximize the differences in the second moment differences
between classes. These features are different than those learned by other
standard techniques like PCA and LDA. SQFA is particularly useful when the
differences in covariances between classes are important for classification.
