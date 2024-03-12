import numpy as np

def lms(patterns, targets, weights, learning_rate, epochs):
    for _ in range(epochs):
        for i in range(patterns.shape[0]):
            a = np.dot(weights[:-1].T, patterns[i]) + weights[-1]
            e = targets[i] - a
            weights[:-1] = weights[:-1] + 2 * learning_rate * e * patterns[i]
            weights[-1] = weights[-1] + 2 * learning_rate * e
    return weights

# Define the patterns and classes
class_A = np.array([[[0],[0]], [[0],[1]], [[1],[0]], [[-1],[-1]]])
class_B = np.array([[[2.1],[0]], [[0], [-2.5]], [[1.6], [-1.5]]])
patterns = np.vstack([class_A, class_B])

t_A = np.ones(class_A.shape[0])
t_B = -1*np.ones(class_B.shape[0])
t = np.concatenate([t_A,t_B])


# Train the ADALINE using the LMS training algorithm
weights = lms(patterns, t, np.array([[0.5],[0.5],[0.5]]), 0.01, 10000)

print("Final weights:", weights)