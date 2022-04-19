#!/usr/bin/env python3
"""
Complete the following source code to plot a stacked bar graph:
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

figure, ax = plt.subplots()
path = ax.scatter(x, y, s=z, c=z)
ax.set_title("Mountain Elevation")
plt.colorbar(path, label="elevation (m)")
plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")
plt.show()

