#!/usr/bin/env python3
"""
Complete the following source code to plot x ↦ y as a scatter plot:
"""
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)


# Plotting
plt.plot(x, y, color='royalblue')
plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.yscale("log")
plt.xlim((0, 28650))
plt.show()
