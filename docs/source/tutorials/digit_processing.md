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

# Digit recognition with SQFA

In this tutorial, we compare SQFA to standard dimensionality reduction
methods in a real-world problem: digit recognition.
We first show that SQFA separates classes better than PCA and LDA
in a complex real-world dataset,
the [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/)
dataset. We then look at a simpler dataset, [MNIST](http://yann.lecun.com/exdb/mnist/),
to explore further the difference between the methods.

## Street View House Numbers (SVHN) dataset

The SVHN dataset consists of images of house numbers taken from Google Street View,
and while it has a similar structure to the MNIST dataset, it is
significantly harder. Let's load the dataset and visualize some of the images.

```{code-cell} ipython3
:tags: [remove-output]

import torch
import matplotlib.pyplot as plt
import torchvision

torch.manual_seed(2)

# Download and load training and test datasets
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

# Convert to PyTorch tensors, average channels and reshape
n_samples, n_channels, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float()
x_train = x_train.mean(dim=1).reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.labels, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float()
x_test = x_test.mean(dim=1).reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.labels, dtype=torch.long)

# Scale data and subtract global mean
def scale_and_center(x_train, x_test):
    std = x_train.std()
    x_train = x_train / (std * n_row)
    x_test = x_test / (std * n_row)
    global_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - global_mean
    x_test = x_test - global_mean
    return x_train, x_test

x_train, x_test = scale_and_center(x_train, x_test)
```

```{code-cell} ipython3
# See how many dimensions, samples and classes we have
print(f"Number of dimensions: {x_train.shape[1]}")
print(f"Number of samples: {x_train.shape[0]}")
print(f"Number of classes: {len(torch.unique(y_train))}")
print(f"Number of test samples: {x_test.shape[0]}")

# Visualize some of the centered images
names = y_train.unique().tolist()
n_classes = len(y_train.unique())
fig, ax = plt.subplots(2, n_classes // 2, figsize=(8, 4))
for i in range(n_classes):
    row = i // 5
    col = i % 5
    ax[row, col].imshow(x_train[y_train == i][20].reshape(n_row, n_col), cmap='gray')
    ax[row, col].axis('off')
    ax[row, col].set_title(names[i], fontsize=10)
plt.tight_layout()
plt.show()
```

We see that we have 10 classes and that the training
data consists of 73257 samples of 1024 dimensions. We will now apply
PCA, LDA and SQFA to learn 9 filters for this dataset.

:::{admonition} Maximum number of filters
A limitation of LDA is that it can learn a maximum of $q-1$ filters, where
$q$ is the number of classes. This is the reason why we learn 9 filters
in this tutorial. PCA and SQFA do not have this limitation.
:::

```{code-cell} ipython3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import sqfa

N_FILTERS = 9

# Train PCA
pca = PCA(n_components=N_FILTERS)
pca.fit(x_train)

# Train LDA
shrinkage = 0.8 # Set to optimize LDA performance and have smoother filters
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
lda = lda.fit(x_train, y_train)

# Train SQFA
# Get noise hyperparameter from PCA variance
x_pca = torch.as_tensor(pca.transform(x_train))
pca_var = torch.var(x_pca, dim=0)
noise = pca_var[2] * 0.1

sqfa_model = sqfa.model.SQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=noise,
  distance_fun=sqfa.distances.affine_invariant
)

sqfa_model.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=False,
)
```

Let's visualize the filters learned by each method.

```{code-cell} ipython3
def plot_filters(filters, title):
    fig, ax = plt.subplots(1, N_FILTERS, figsize=(10, 3))
    for i in range(N_FILTERS):
        ax[i].imshow(filters[i].reshape(n_row, n_col), cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"Filter {i+1}")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# PCA filters
plot_filters(pca.components_, "PCA filters")

# LDA filters
plot_filters(lda.coef_, "LDA filters")

# SQFA filters
plot_filters(sqfa_model.filters.detach(), "SQFA filters")
```

The features learned by the three models look different. PCA filters look
smooth, and some structure related to the digits is visible, but this
structure does not seem visually diverse (most filters look like
3's and 8's).

LDA filters look less smooth, and although they also seem to capture
some structure related to the digits, this structure consists of
some local feature patterns.

SQFA filters, in contrast, look smooth[^1] and capture a set of features that
look like digits.

Lets evaluate the performance of the filters in separating the classes
by using a quadratic classifier, Quadratic Discriminant Analysis (QDA).
QDA fits a Gaussian distribution (mean and covariance) to each class and uses
the Bayes rule to classify samples. Both the first order (class means) and
second order (class covariances) are used to classify samples.

```{code-cell} ipython3
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_qda_accuracy(x_train, y_train, x_test, y_test):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    y_pred = qda.predict(x_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

model_list = [pca, lda, sqfa_model]
accuracies = []

for model in model_list:
    with torch.no_grad():
        x_train_features = model.transform(x_train)
        x_test_features = model.transform(x_test)
        accuracy = get_qda_accuracy(x_train_features, y_train, x_test_features, y_test)
        accuracies.append(accuracy * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), ["PCA", "LDA", "SQFA"])
plt.title(f"QDA accuracy for different feature extraction methods")
plt.ylabel("QDA Accuracy (%)")
plt.xlabel("Features")
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.tight_layout()
plt.show()
```

We see that for this problem, SQFA outperforms PCA and LDA by a large margin
in terms of classification accuracy. This indicates that the classes are
better separated in the feature space learned by SQFA. Similar results
are obtained when comparing SQFA to other more sophisticated dimensionality
reduction methods (Independent Component Analysis, Factor Analysis).

Why is SQFA so much better than PCA and LDA at separating the classes?
For the case of PCA, separating classes is not the main goal of the method,
so it might not be surprising that SQFA features do better. For LDA,
linear separability of the classes in the main goal. LDA is expected
to not perform as well as SQFA when the classes are not linearly separable,
i.e. when the class means are not far apart. Looking at the SVHN stimuli
above, we can notice that there is considerable nuisance variability
in the digits. An important source of variability is that the digits
can have different polarities, i.e. they can be dark digits on a dark background
or light digits on a dark background. This is going to diminish the
differences in the class means.

## MNIST dataset

The MNIST dataset is a simpler dataset than SVHN. An important difference
is that in MNIST the digits are more uniformly centered and scaled,
and they always have the same polarity: white digits on a black background.
This means that first-order differences between classes are larger, and
even single pixels can do a good job at separating pairs of classes.

Let's compare PCA, LDA and SQFA in this dataset.

```{code-cell} ipython3
:tags: [remove-output]
# Load MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

n_samples, n_row, n_col = trainset.data.shape
n_dim = n_row * n_col
x_train = trainset.data.reshape(-1, n_dim).float()
y_train = trainset.targets
x_test = testset.data.reshape(-1, n_dim).float()
y_test = testset.targets

# Scale data and subtract global mean
x_train, x_test = scale_and_center(x_train, x_test)
```

```{code-cell} ipython3
# See how many dimensions, samples and classes we have
print(f"Number of dimensions: {x_train.shape[1]}")
print(f"Number of samples: {x_train.shape[0]}")
print(f"Number of classes: {len(torch.unique(y_train))}")
print(f"Number of test samples: {x_test.shape[0]}")

# Visualize some of the centered images
names = y_train.unique().tolist()
n_classes = len(y_train.unique())
fig, ax = plt.subplots(1, n_classes, figsize=(10, 2))
for i in range(n_classes):
    ax[i].imshow(x_train[y_train == i][20].reshape(n_row, n_col), cmap='gray')
    ax[i].axis('off')
    ax[i].set_title(names[i], fontsize=10)
plt.tight_layout()
plt.show()
```

We see that the MNIST stimuli are a lot cleaner than the SVHN stimuli.
Let's now apply PCA, LDA and SQFA to learn 9 filters for this dataset
and visualize the filters.

```{code-cell} ipython3
# Train PCA
pca = PCA(n_components=N_FILTERS)
pca.fit(x_train)

# Train LDA
shrinkage = 0.8 # Set to optimize LDA performance and have smoother filters
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
lda = lda.fit(x_train, y_train)

# Train SQFA
# Get noise hyperparameter from PCA variance
x_pca = torch.as_tensor(pca.transform(x_train))
pca_var = torch.var(x_pca, dim=0)
noise = pca_var[2] * 0.5

sqfa_model = sqfa.model.SQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=noise,
  distance_fun=sqfa.distances.affine_invariant
)

sqfa_model.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=False
)

# PCA filters
plot_filters(pca.components_, "PCA filters")

# LDA filters
plot_filters(lda.coef_, "LDA filters")

# SQFA filters
plot_filters(sqfa_model.filters.detach(), "SQFA filters")
```

We see that now the LDA filters capture clear digit structures. SQFA filters
capture some digit structure, but to a lesser extent than LDA filters.
Let's evaluate the performance of the filters in separating the classes.

```{code-cell} ipython3
model_list = [pca, lda, sqfa_model]
accuracies = []

for model in model_list:
    with torch.no_grad():
        x_train_features = model.transform(x_train)
        x_test_features = model.transform(x_test)
        accuracy = get_qda_accuracy(x_train_features, y_train, x_test_features, y_test)
        accuracies.append(accuracy * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), ["PCA", "LDA", "SQFA"])
plt.title(f"QDA accuracy for different feature extraction methods")
plt.ylabel("QDA Accuracy (%)")
plt.xlabel("Features")
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.tight_layout()
plt.show()
```

We see that, for this problem, LDA and SQFA features perform similarly,
although LDA features perform slightly better. This means that the
classes are well separated linearly, and in this case SQFA may not
separate the data better than LDA.[^2]

## Random polarity MNIST

To finalize the analysis, lets create a new version of the MNIST dataset
where the polarity of the digits is random. This will make the dataset more
complex, and reduce the first order differences between classes.

```{code-cell} ipython3
# Get un-preprocessed MNIST again
x_train = trainset.data.reshape(-1, n_dim).float()
y_train = trainset.targets
x_test = testset.data.reshape(-1, n_dim).float()
y_test = testset.targets

# Randomly revert the polarity for half of the images
inds = torch.randint(0, 2, (len(y_train),))
x_train[inds==0] = - x_train[inds==0]
inds = torch.randint(0, 2, (len(y_test),))
x_test[inds==0] = - x_test[inds==0]

# Scale data and subtract global mean
x_train, x_test = scale_and_center(x_train, x_test)

# Visualize some of the centered images
names = y_train.unique().tolist()
n_classes = len(y_train.unique())
min = x_train.min()
max = x_train.max()

fig, ax = plt.subplots(1, n_classes, figsize=(10, 2))
for i in range(n_classes):
    ax[i].imshow(x_train[y_train == i][20].reshape(n_row, n_col), cmap='gray',
                 vmin=min, vmax=max)
    ax[i].axis('off')
    ax[i].set_title(names[i], fontsize=10)
plt.tight_layout()
plt.show()
```

Let's now apply PCA, LDA and SQFA to learn 9 filters for this dataset.

```{code-cell} ipython3
# Train PCA
pca = PCA(n_components=N_FILTERS)
pca.fit(x_train)

# Train LDA
shrinkage = 0.7 # Set to optimize LDA performance and have smoother filters
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
lda = lda.fit(x_train, y_train)

# Train SQFA
# Get noise hyperparameter from PCA variance
x_pca = torch.as_tensor(pca.transform(x_train))
pca_var = torch.var(x_pca, dim=0)
noise = pca_var[2] * 0.5

sqfa_model = sqfa.model.SQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=noise,
  distance_fun=sqfa.distances.affine_invariant
)

sqfa_model.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=False
)

# PCA filters
plot_filters(pca.components_, "PCA filters")

# LDA filters
plot_filters(lda.coef_, "LDA filters")

# SQFA filters
plot_filters(sqfa_model.filters.detach(), "SQFA filters")
```

We see that LDA filters capture some digit structures although this is
embedded in a lot of noise. SQFA filters, again, capture only some
digit structure. Let's evaluate the performance of the filters in separating
the classes.

```{code-cell} ipython3
model_list = [pca, lda, sqfa_model]
accuracies = []

for model in model_list:
    with torch.no_grad():
        x_train_features = model.transform(x_train)
        x_test_features = model.transform(x_test)
        accuracy = get_qda_accuracy(x_train_features, y_train, x_test_features, y_test)
        accuracies.append(accuracy * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(6, 4))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), ["PCA", "LDA", "SQFA"])
plt.title(f"QDA accuracy for different feature extraction methods")
plt.ylabel("QDA accuracy (%)")
plt.xlabel("Features")
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.tight_layout()
plt.show()
```

We see that, for the dataset of MNIST with random polarity, SQFA features
outperform LDA features by a large margin. This is consistent with the
fact that the classes should have low linear separability in this dataset.
SQFA features also outperform PCA features, although by a smaller margin.
This result shows that feature learning with SQFA can be more invariant
that with LDA, in this particular case shown with the example of
digit polarity variability in MNIST.

In summary, we have shown that SQFA can outperform PCA and LDA in separating
classes in a complex real-world dataset, the SVHN dataset. In simpler datasets,
with more linearly separable classes, SQFA may not outperform LDA.
Thus, the choice of dimensionality reduction method should be made
considering the complexity of the dataset and the goal of the analysis.


[^1]: We note that the smoothness of both LDA and SQFA filters depends on the
hyperparameters. For LDA, we obtain smoother filters with higher shrinkage parameter,
and for SQFA we obtain smoother filters with higher noise parameter. However,
in general LDA filters tend to be less smooth than PCA and SQFA filters.
[^2]: Note that the noise hyperparameter is different in the SVHN and MNIST datasets.
Performance of SQFA features learned for the MNIST dataset is sensitive to this
parameter. With lower noise values, SQFA features perform considerably worse
than LDA features.
