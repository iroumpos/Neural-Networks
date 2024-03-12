#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import time

# Function to perform convolution using the first approach
def convolve_horizontal_strip(image, kernel):
    # Assuming kernel size is odd
    k = kernel.shape[0]
    padding = k // 2
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(padding, image.shape[1] - padding):
            output[i, j] = np.sum(image[i, j - padding: j + padding + 1] * kernel)
    return output

# Function to perform convolution using the second approach
def convolve_wider_strip(image, kernel, delta):
    # Assuming kernel size is odd
    k = kernel.shape[0]
    padding = k // 2
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(padding, image.shape[1] - padding, delta):
            output[i, j:j+delta] = np.sum(image[i, j - padding: j + padding + 1] * kernel)
    return output

# Generate sample image
image_size = 228
sample_image = np.random.randint(0, 255, size=(image_size, image_size))

# Define kernels
kernel_3x3 = np.random.rand(3, 3)
kernel_7x7 = np.random.rand(7, 7)
kernel_11x11 = np.random.rand(11, 11)

# Measure execution time for 3x3 kernel
start_time = time.time()
_ = convolve_horizontal_strip(sample_image, kernel_3x3)
horiz_strip_time = time.time() - start_time

start_time = time.time()
_ = convolve_wider_strip(sample_image, kernel_3x3, delta=3)
wider_strip_time = time.time() - start_time

print("Execution time for 3x3 kernel (horizontal strip):", horiz_strip_time)
print("Execution time for 3x3 kernel (wider strip):", wider_strip_time)

# Measure execution time for 7x7 kernel
start_time = time.time()
_ = convolve_horizontal_strip(sample_image, kernel_7x7)
horiz_strip_time = time.time() - start_time

start_time = time.time()
_ = convolve_wider_strip(sample_image, kernel_7x7, delta=7)
wider_strip_time = time.time() - start_time

print("Execution time for 7x7 kernel (horizontal strip):", horiz_strip_time)
print("Execution time for 7x7 kernel (wider strip):", wider_strip_time)

# Measure execution time for 11x11 kernel
start_time = time.time()
_ = convolve_horizontal_strip(sample_image, kernel_11x11)
horiz_strip_time = time.time() - start_time

start_time = time.time()
_ = convolve_wider_strip(sample_image, kernel_11x11, delta=11)
wider_strip_time = time.time() - start_time

print("Execution time for 11x11 kernel (horizontal strip):", horiz_strip_time)
print("Execution time for 11x11 kernel (wider strip):", wider_strip_time)


# In[ ]:




