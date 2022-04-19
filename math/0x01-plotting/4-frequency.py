#!/usr/bin/env python3
"""
Complete the following source code to plot x ↦ y1 and x ↦ y2 as line graphs:
"""
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

x = np.linspace(0, 100, 11)

# Plotting
# title of graph
plt.title("Project A")
# labels of axes data
plt.xlabel("Grades")
plt.ylabel("Number of Students")
# show all numbers in x axis
plt.xticks(x)
# limit the numbers that show in both axes
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.hist(student_grades, bins=x, edgecolor="black")
plt.show()
