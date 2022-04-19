#!/usr/bin/env python3
"""
Complete the following source code to plot a stacked bar graph:
"""
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(5)

fruit = np.random.randint(0, 20, (4, 3))

persons = ["Farrah", "Fred", "Felicia"]
width = 0.5
figure, ax = plt.subplots()
ax.set_title("Number of Fruit per Person")

plt.ylabel("Quantity of Fruit")
plt.yticks(np.arange(0, 81, 10))
plt.legend(loc="upper right")
plt.ylim((0.0, 80.0))

# bars of fruit
ax.bar(persons, fruit[0], width, color='r', label="apples")
ax.bar(persons, fruit[1], width, bottom=fruit[0], color='yellow',
       label="bananas")
ax.bar(persons, fruit[2], width, bottom=fruit[1] + fruit[0],
       color='#ff8000', label="oranges")
ax.bar(persons, fruit[3], width, bottom=fruit[2] + fruit[1] + fruit[0],
       color='#ffe5b4', label="peaches")
plt.legend()

plt.show()
