# swfilter
A Python implementation of the Sliced-Wasserstein Filter developed by Julien Pallage and Antoine Lesage-Landry.

---

## Introduction:
In this work, we present a new unsupervised anomaly (outlier) detection (AD) method using the sliced-Wasserstein metric. This filtering technique is conceptually interesting for integration in MLOps pipelines deploying trustworthy machine learning models in critical sectors like energy. We also propose an approximation of our methodology using a Fast Euclidian variation. The code is written to respect scikit-learn's API and be called similarly to other scikit-learn AD methods, e.g., Isolation Forest, Local Outlier Factor.

## How it is made:
We use the Python implementation of the sliced-Wasserstein distance from the library `POT` and use a voting system to label candidate samples as outliers or inliers and we use `joblib` to parallelize the procedure.

## How to use it:

```python
from swfilter import SlicedWassersteinFilter
eps = 0.01 # the threshold of the SW distance
n = 30 # the number of voters
n_projections = 50 # the number of projections used in the SW computations
p = 0.6 # the threshold percentage of voters required to label as outlier
n_jobs = -1 # the number of workers to call in the parallelization (-1 = max)

model = SlicedWassersteinFilter(eps=eps, n=n, n_projections=n_projections, p=p, n_jobs=n_jobs, swtype='original')
preds, vote = model.fit_predict(dataset)

mask = preds == 1
filtered_dataset = dataset[mask]
```

## Install:
Coming soon!

### Tutorial:

See our tutorial page!

[link](https://github.com/jupall/swfilter/blob/main/experiments/tutorial.ipynb)

## Cite our work and read our paper:

Coming soon!
