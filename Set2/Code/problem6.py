#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# Activation function (tanh)
def tanh(x):
    return np.tanh(x)

# Derivative of the activation function
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Define the neural network parameters
w = np.array([1, 0.5, -1])  # Weights
p = np.array([0, 0, 1])      # Input
bias = 1                     # Bias
a = 1                        # Learning rate
t = 0.75                     # Target label

# Forward pass
net_input = np.dot(w, p) + bias
output = tanh(net_input)

def tanh_derivative(x, t, output, p):
    return np.array([-2 * (t - output) * (1 - np.tanh(x) ** 2) * p[0],
                     -2 * (t - output) * (1 - np.tanh(x) ** 2) * p[1],
                     -2 * (t - output) * (1 - np.tanh(x) ** 2) * p[2]])



delta_weights =tanh_derivative(net_input, t, output, p)


# Update weights
w += delta_weights

# Update bias
db=-2 * (t - output) * (1 - np.tanh(net_input) ** 2)

bias+=db


print("Updated weights after 1 iteration:", w)
print("Updated bias after 1 iteration:",bias)

