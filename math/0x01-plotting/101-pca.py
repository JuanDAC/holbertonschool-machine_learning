#!/usr/bin/env python3
"""
Complete the following source code to plot a stacked bar graph:
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

figure = plt.figure()
axes = Axes3D(figure)

x, y, z = np.stack(pca_data, axis=-1)
axes.scatter(x, y, z, c=labels, cmap=cm.plasma)

plt.xlabel("U1")
plt.ylabel("U2")
plt.zlabel("U3")
plt.title("PCA of Iris Dataset")
plt.show()
