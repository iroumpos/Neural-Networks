#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.signal import convolve2d

# Define the input matrix and kernel
input_matrix = np.array([[20,35,35,35,35,20],
                         [29,46,44,42,42,27],
                         [16,25,21,19,19,19],
                         [66,120,116,154,114,62],
                         [74,216,174,252,172,112],
                         [70,210,170,250,170,110]])

kernel = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]])

# Perform convolution with stride=1 using scipy's convolve2d
result_matrix = convolve2d(input_matrix, kernel, mode='valid')

# Print the result
print("Result Matrix:")
print(result_matrix)


# In[10]:


from skimage.measure import block_reduce

stride = 2
block_size = (2, 2)

# Apply max pooling using scipy's maximum_filter
max_pooled_output = block_reduce(result_matrix, block_size=block_size, func=np.max)

# Print the max-pooled result
print("Max Pooled Output:")
print(max_pooled_output)

