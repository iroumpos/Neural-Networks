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

# # $F(x) = c + d^T \cdot x + \frac{1}{2} \cdot x^T \cdot A \cdot x$



import numpy as np 
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show_config
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import exp,arange






# # Data

# +


list1 = [[1],
         [2]]
list2 = [[-2],
         [1]]

p1 = np.array(list1)
p2 = np.array(list2)
t1 = -1
t2 = 1

prob_t1 = 0.5
prob_t2 = 0.5

# -



# # Calculate c 

c = prob_t1 * (t1 ** 2) + prob_t2 * (t2 ** 2)
c



# # Calculate h

h = prob_t1 * t1 * p1 + prob_t2 * t2 * p2
h



# # Calculate R

R = prob_t1 * p1 * np.transpose(p1) + prob_t2 * p2 * np.transpose(p2)
R



# # Mean square error performance index 
# ## $F(x) = c + d^T \cdot x + \frac{1}{2} \cdot x^T \cdot A \cdot x$

# +
from sympy import symbols, Matrix
w11, w12 = symbols('w11 w12')

F = 1 + 3 * w11 + w12 + 2.5 * w11 ** 2 + 2.5 * w12 ** 2
F
# -



# # Contour plot

# +
f = lambda x,y: 1 + 3 * x + y + 2.5 * x ** 2 + 2.5 * y ** 2

x = np.linspace(-3.5, 3.5, 200)
y = np.linspace(-3.5, 3.5, 200)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, 20, cmap='RdGy');
plt.colorbar();
# Add labels and title
plt.xlabel(r'$w_{1,1}$')
plt.ylabel(r'$w_{1,2}$')
plt.title('Contour Plot of MSE Performance Index')

# -





# # Optimal decision boundary



R_inverse = np.linalg.inv(R)
x_star = np.matmul(R_inverse,h)
x_star







# # LMS

# +
vectors = np.hstack((p1, p2))
vectors

targets = np.array([t1, t2])
targets
# -



# # Initialize weights

weights = np.array([3, 1])
weights



# # Learning rate

lr = 0.025



# # Implementation of LMS 



# +
trajectory_x = [weights[0]]
trajectory_y = [weights[1]]

for k in range(50):
    for j in range(len(targets)):
        curr_vector = vectors[:, j]
        curr_target = targets[j]
    
        a = np.matmul(weights,curr_vector)
        error = curr_target - a
        weights = weights + 2 * lr * error * np.transpose(curr_vector)
        trajectory_x.append(weights[0])
        trajectory_y.append(weights[1])
weights
# -



# # Trajectory of weights update on the contour plot



# +
plt.contour(X, Y, Z, 20, cmap='RdGy');
plt.colorbar();

# Plot the trajectory on top of the contour plot
plt.plot(trajectory_x, trajectory_y, marker='o', color='red', label='Trajectory')
plt.legend()
plt.title('Contour Plot with Trajectory')
    
    
# Add labels and title
plt.xlabel(r'$w_{1,1}$')
plt.ylabel(r'$w_{1,2}$')
plt.title('Contour Plot of MSE Performance Index')
plt.show()
# -






















