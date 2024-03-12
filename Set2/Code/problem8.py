import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the function F(w)
def F(w):
    return 0.1 * (w[0] + w[1])**2 + 2 * (w[0] - w[1])**2

# Define the gradient of F(w)
def grad_F(w):
    return torch.tensor([0.2 * (w[0]+w[1]) + 4*(w[0] + w[1]), 0.2 * (w[0]+w[1]) - 4*(w[0] - w[1])])

# Define the initial point and initialize Adadelta optimizer
w = torch.tensor([3.0, 2.0], requires_grad=True)
optimizer = optim.Adadelta([w], lr=3)

# Lists to store trajectory points
trajectory_x = []
trajectory_y = []

# Number of iterations
num_iterations = 10000

# Optimization loop
for i in range(num_iterations):
    # Store current point
    trajectory_x.append(w[0].item())
    trajectory_y.append(w[1].item())
    
    # Take a step using Adadelta
    optimizer.zero_grad()
    loss = F(w)
    loss.backward()
    optimizer.step()

# Plot contour plot of F(x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = F(torch.tensor([X, Y])).numpy()

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50)
plt.plot(trajectory_x, trajectory_y, marker='o', color='r')
plt.xlabel('w[0]')
plt.ylabel('w[1]')
plt.title('Contour Plot of F(w) with Adadelta Trajectory')
plt.colorbar(label='F(w)')
plt.grid(True)
plt.savefig('p8_c_2.png')
plt.show()

