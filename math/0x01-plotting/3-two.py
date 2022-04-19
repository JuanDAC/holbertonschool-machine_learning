#!/usr/bin/env python3
"""
Complete the following source code to plot x ↦ y1 and x ↦ y2 as line graphs:
"""
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)


# Plotting
plt.plot(x, y1, 'r--', label='C-14')
plt.plot(x, y2, color='green', label='Ra-226')
plt.title("Exponential Decay of Radioactive Elements")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.xlim((0, 20000))
plt.ylim((0, 1))
plt.legend()
plt.show()
