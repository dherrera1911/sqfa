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

# SQFA vs other techniques - Toy Problem

In this tutorial we build a simple toy problem to illustrate the types of features
that are learnt by SQFA, and how they compare to other standard feature
learning techniques.

## SQFA vs PCA in the 0-mean case

To compare the methods, we will generate a toy problem with 3 classes over a 4
dimensional space. The first two and the last two dimensions of the data space
will vary in different ways, and we will test what dimensions are emphasized by
SQFA vs PCA and Linear Discriminant Analysis (LDA).

SQFA finds the features that maximize the difference in second-order statistics
between the classes. PCA, on the other hand, finds the features that
maximize the variance of the data. In our first example, we will generate
statistics for 3 classes with the following properties:

- Dimensions 1 and 2 have high variance, but the same covariance
  structure for all classes.
- Dimensions 3 and 4 have lower variance, but the covariances are different
  for each class.
- Dimensions 1 and 2 are uncorrelated to dimensions 3 and 4.
- The means are 0 for all dimensions and classes.

To achieve this, we will generate an initial covariance matrix, and we will
progressively rotate the covariance matrix for dimensions 3 and 4 for each
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

Let's generate the covariance matrices and visualize them:

```{code-cell} ipython3
# Base diagonal covariance
variances = torch.tensor([1.0, 1.0, 0.8, 0.02])
base_cov = torch.diag(variances)

angles = torch.as_tensor([15, 45, 70])
class_covariances = make_rotated_classes(base_cov, angles)

# Plot the covariances
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sqfa.plot.statistics_ellipses(ellipses=class_covariances, dim_pair=(0, 1), ax=ax[0])
sqfa.plot.statistics_ellipses(ellipses=class_covariances, dim_pair=(2, 3),
                              ax=ax[1], legend_type='discrete')
plt.show()
```

The first plot shows the covariance for dimensions 1 and 2, which is the same
for all classes. The second plot shows how the covariance for dimensions 3 and 4
rotates for each class.

Let's learn 2 features using SQFA and PCA for this problem. To
obtain the PCA features we average the covariance matrices across classes
to obtain the dataset covariance
and [compute its eigenvectors.](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#covariance-matrix)

```{code-cell} ipython
# Learn SQFA filters
model = sqfa.model.SQFA(
  input_covariances=class_covariances,
  feature_noise=0.001,
  n_filters=2
)
model.fit(epochs=20)
sqfa_filters = model.filters.detach()

# Learn PCA filters
average_cov = torch.mean(class_covariances, dim=0)
eigval, eigvec = torch.linalg.eigh(average_cov)
pca_filters = eigvec[:, -2:].T
```

Let's plot the filters learned by SQFA and PCA:

```{code-cell} ipython
# Plot the filters
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
x = torch.arange(4) + 1
for f in range(2):
    ax[0].plot(x, sqfa_filters[f])
    ax[1].plot(x, pca_filters[f])
ax[0].set_title('SQFA filters')
ax[1].set_title('PCA filters')
ax[0].set_xlabel('Dimension')
ax[1].set_xlabel('Dimension')
ax[0].set_ylabel('Weight')
plt.tight_layout()
plt.show()

```

We see that the filters learned by SQFA put more weight on the dimensions
with differences in covariances, while PCA puts more weight on the dimensions
with higher variance. Let's next compute and plot the statistics of both
types of features.

```{code-cell} ipython
# Get statistics and plot
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
pca_covariances = torch.einsum('ij,njk,kl->nil', pca_filters, class_covariances, pca_filters.T)

kwargs = {
    'bbox_to_anchor': (1.05, 1),
    'loc': 'upper left',
    'borderaxespad': 0.0,
    'title': 'Class',
} # Legend arguments

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances, ax=ax[0])
sqfa.plot.statistics_ellipses(ellipses=pca_covariances, ax=ax[1], legend_type='discrete', **kwargs)
ax[0].set_title('SQFA features')
ax[1].set_title('PCA features')
ax[0].set_xlabel('Feature 1')
ax[0].set_xlabel('Feature 2')
ax[1].set_xlabel('Feature 1')
ax[1].set_xlabel('Feature 2')
plt.show()
```

We see that while the SQFA features maintain the differences in covariances between
classes, PCA features miss this information. With the use of an appropriate
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
class_means = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0]])
class_means = class_means * 0.5

# Plot the new distributions
dim_pairs = [(0, 1), (2, 3)]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i, dim_pair in enumerate(dim_pairs):
    sqfa.plot.statistics_ellipses(ellipses=class_covariances,
                                  centers=class_means,
                                  dim_pair=dim_pair,
                                  ax=ax[i])
    sqfa.plot.scatter_data(data=class_means,
                           labels=torch.arange(3),
                           dim_pair=dim_pair,
                           ax=ax[i])
    ax[i].set_xlabel(f'Dimension {dim_pair[0]+1}')
    ax[i].set_ylabel(f'Dimension {dim_pair[1]+1}')
plt.tight_layout()
plt.show()
```

We can see that the means of the classes are different in the first two dimensions.
Let's learn the filters with SQFA and LDA now and compare the results. First, we
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
model = sqfa.model.SQFA(
  input_covariances=second_moments,
  feature_noise=0.001,
  n_filters=2
)
loss, time = model.fit(epochs=20)
sqfa_filters = model.filters.detach()
```

Like in the previous example, we plot the filters learned by SQFA and LDA:

```{code-cell} ipython
# Plot the filters
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
x = torch.arange(4) + 1
for f in range(2):
    ax[0].plot(x, sqfa_filters[f])
    ax[1].plot(x, lda_filters[f])
ax[0].set_title('SQFA filters')
ax[1].set_title('LDA filters')
ax[0].set_xlabel('Dimension')
ax[1].set_xlabel('Dimension')
ax[0].set_ylabel('Weight')
plt.tight_layout()
plt.show()
```

We see that, as expected, LDA puts more weight on the dimensions with differences
in the means, while SQFA puts more weight on the dimensions with differences in
the covariances. Let's plot the statistics of the features learned by both methods.

```{code-cell} ipython
# Get the means and covariances for the new features
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
lda_covariances = torch.einsum('ij,njk,kl->nil', lda_filters, class_covariances, lda_filters.T)
sqfa_means = class_means @ sqfa_filters.T
lda_means = class_means @ lda_filters.T

kwargs = {
    'bbox_to_anchor': (1.05, 1),
    'loc': 'upper left',
    'borderaxespad': 0.0,
    'title': 'Class',
} # Legend arguments

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

# SQFA statistics
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances,
                              centers=sqfa_means, ax=ax[0])
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax[0])
# LDA statistics
sqfa.plot.statistics_ellipses(ellipses=lda_covariances, centers=lda_means,
                              ax=ax[1], legend_type='discrete', **kwargs)
sqfa.plot.scatter_data(data=lda_means, labels=torch.arange(3), ax=ax[1])

ax[0].set_title('SQFA features')
ax[1].set_title('LDA features')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')
plt.show()
```

We see that the distribution of LDA features have different means, while
the distribution of SQFA features have the same means but different covariances.
Which features are better for classification will depend on the specifics of
the class distributions, as well as the classifier used.


## SQFA is sensible to covariances and means

In the previous example we showed that SQFA prioritized the differences in
covariances over the differences in means. However, this is not always the case.
Particularly, note that we fitted SQFA using the second moment matrices of the
classes, which for a given class $i$ are given
by $\Psi_i = \Sigma_i + \mu_i \mu_i^T$. Thus, the second moments of a class will
be influenced by both the covariance matrix and the mean of the class.

We can see this by modifying the toy example above to have larger differences
in the means between the first two dimensions. Let's make the class means
more different and plot the new distributions.

```{code-cell} ipython
class_means = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0]])
class_means = class_means * 5.0

# Plot the new distributions
dim_pairs = [(0, 1), (2, 3)]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i, dim_pair in enumerate(dim_pairs):
    sqfa.plot.statistics_ellipses(ellipses=class_covariances,
                                  centers=class_means,
                                  dim_pair=dim_pair,
                                  ax=ax[i])
    sqfa.plot.scatter_data(data=class_means,
                           labels=torch.arange(3),
                           dim_pair=dim_pair,
                           ax=ax[i])
    ax[i].set_xlabel(f'Dimension {dim_pair[0]+1}')
    ax[i].set_ylabel(f'Dimension {dim_pair[1]+1}')
plt.tight_layout()
plt.show()
```

Let's recompute the second moments and learn the filters with SQFA again.

```{code-cell} ipython
# Get the new second moment matrices
mean_outer_prod = torch.einsum('ij,ik->ijk', class_means, class_means)
second_moments = class_covariances + mean_outer_prod

# Learn SQFA
model = sqfa.model.SQFA(
  input_covariances=second_moments,
  feature_noise=0.001,
  n_filters=2
)
loss, time = model.fit(epochs=20)
sqfa_filters = model.filters.detach()
```

Let's plot the filters learned by SQFA and the statistics of the features.

```{code-cell} ipython
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Plot the filters
x = torch.arange(4) + 1
for f in range(2):
    ax[0].plot(x, sqfa_filters[f])
ax[0].set_title('SQFA filters')
ax[0].set_xlabel('Dimension')
ax[0].set_ylabel('Weight')

# Get the means and covariances for the new features
sqfa_covariances = torch.einsum('ij,njk,kl->nil', sqfa_filters, class_covariances, sqfa_filters.T)
sqfa_means = class_means @ sqfa_filters.T

# Plot the statistics
sqfa.plot.statistics_ellipses(ellipses=sqfa_covariances,
                              centers=sqfa_means, ax=ax[1])
sqfa.plot.scatter_data(data=sqfa_means, labels=torch.arange(3), ax=ax[1])
ax[1].set_title('SQFA features')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')
plt.tight_layout()
plt.show()
```
  
We see that now the filters learned by SQFA put more weight on the dimensions
with differences in the means, and that this is reflected in the
feature statistics. This example illustrates that SQFA is sensitive to both
the covariances and the means of the classes, and that the features learned
will depend on the specifics of the class distributions. Note that this is
the case if we use the second moment matrices as input to SQFA. If we use
only the covariance matrices, SQFA will only be sensitive to the differences
in the covariances between classes.

## Conclusion

SQFA is a feature learning technique that is sensitive to the differences in
second moments between classes. It learns different features than other
standard techniques like PCA and LDA, and can be particularly useful when
the differences in covariances between classes are important for classification.

