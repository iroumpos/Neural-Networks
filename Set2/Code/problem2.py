#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sympy as sym
import numpy as np

w1, w2 = sym.symbols('w1 w2')

# Initial values
w = np.array([3, 3])  # 1x1 matrix

# Epsilon for convergence
epsilon = 1e-6

# Function
f = w1**2 + w2**2 + (0.5*w1 + w2)**2 + (0.5*w1 + w2)**4

# The expression of gradient
df_w1 = sym.diff(f, w1)
df_w2 = sym.diff(f, w2)

# Lambdify the gradient expressions
gradient_func = sym.lambdify((w1, w2), [df_w1, df_w2])
print(df_w1)

# The expressions of Hessian Matrix
a_00 = sym.diff(df_w1, w1)
a_01 = sym.diff(df_w1, w2)
a_10 = sym.diff(df_w2, w1)
a_11 = sym.diff(df_w2, w2)

print(a_00)
print(a_01)
print(a_10)
print(a_11)

# Lambdify the Hessian expressions
hessian_func = sym.lambdify((w1, w2), [[a_00, a_01], [a_10, a_11]])

while True:
    # Calculate gradient and Hessian using lambdified functions
    gradient = np.array(gradient_func(w[0], w[1]))  # 1x1 matrix
    hessian = np.array(hessian_func(w[0], w[1]))    # 2x2 matrix

    # Inverted hessian
    inv_hessian = np.linalg.inv(hessian)

    # Direction
    s = np.dot(-inv_hessian, gradient)

    # Optimal l for our problem is constant
    l = 1

    # Next point
    next_w = np.squeeze(np.add(w, np.dot(l, s)))  # Remove single-dimensional entries

    # Norm difference
    norm_diff = np.linalg.norm(next_w - w)
    print(w)
    # Update the previous value
    w = next_w

    if norm_diff < epsilon:
        print("Converged!")
        break

print(w)


# In[ ]:




