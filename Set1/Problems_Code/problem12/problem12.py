# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ! jupytext --to py basic_examples_usage_autograd.ipynb

# +
from numpy import *
import math
import matplotlib.pyplot as plt

g = lambda w: 0*w
f = lambda z: (z-2)/3
h = lambda m: 1 - (m-3)/4

x1 = linspace(0, 2, 400)
x2 = linspace(2, 4.14, 400)
x3 = linspace(4.14, 7, 400)
x4 = linspace(7, 10, 400)

g_values = g(x1)
f_values = f(x2)
h_values = h(x3)
g_values2 = g(x4)




plt.figure(figsize=(8,6))
plt.plot(x1, g_values, color = 'b', label = '$H_1 = 0$')
plt.plot(x2, f_values, color = 'magenta', label = r'$H_2 = \frac{x-2}{3}$')
plt.plot(x3, h_values, color = 'red', label = r'$H_3 = 1 - \frac{x-3}{4}$')
plt.plot(x4, g_values2, color = 'b', label = '$H_4 = 0$')
# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
#plt.title(r'$\neg(A(X) U B(X))$')
plt.title(r'$\overline{A(X) \cup B(X)}$')
plt.legend(fontsize=13)
plt.show()
# -













































# $\neg(A(X) U B(X))$





















#


















