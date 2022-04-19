#!/usr/bin/env python3
"""
Complete the following source code to plot x ↦ y as a scatter plot:
"""
import matplotlib.pyplot as plt
import numpy as np


mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180


# Plotting
plt.scatter(x, y, color='magenta')
plt.title("Men's Height vs Weight")
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.show()
