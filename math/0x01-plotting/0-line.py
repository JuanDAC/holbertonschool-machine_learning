#!/usr/bin/env python3
"""
Progress vs Score Task Body Complete the following source code to plot y
as a line graph
y should be plotted as a solid red line The x-axis should
range from 0 to 10 plot
"""
import matplotlib.pyplot as plt
import numpy as np


# Values
def rang():
    """ Rabge of point of x axis"""
    return 11


def from_x():
    """ First point of x axis"""
    return 0.0


def to_x():
    """ First point of x axis"""
    return 10.0


y = np.arange(0, 11) ** 3
x = np.linspace(from_x(), to_x(), num=rang())

# Plotting
plt.plot(x, y, color='r')
plt.xlim((0.0, 10.0))
plt.show()
