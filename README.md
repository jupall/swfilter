# swfilter
A Python implementation of the Sliced-Wasserstein Filter developed by Julien Pallage and Antoine Lesage-Landry.

---

## Introduction:
In this work, we present a new unsupervised anomaly (outlier) detection (AD) method using the sliced-Wasserstein metric. This filtering technique is conceptually interesting for integration in MLOps pipelines deploying trustworthy machine learning models in critical sectors like energy. We also propose an approximation of our methodology using a Fast Euclidian variation. The code is written to respect scikit-learn's API and be called similarly to other sciki-learn AD methods, e.g., Isolation Forest, Local Outlier Factor.

## How it is made:
We use the Python implementation of the sliced-Wasserstein distance from the library `POT` and use a voting system to label candidate samples as outliers or inliers.

## How to use it:

See our tutorial page!

link

## Cite our work and read our paper:

Coming soon!
