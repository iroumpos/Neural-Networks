import matplotlib.pyplot as plt

def henon_map(x, y, a, b):
    xn = 1 - a * x**2 + b * y
    yn = x
    return xn, yn

def normalize(value, lower_limit, upper_limit):
    return max(min(value, upper_limit), lower_limit)

def generate_trajectory(x0, y0, a, b, iterations):
    x_vals = [x0]
    y_vals = [y0]
    for i in range(iterations):
        x, y = henon_map(x_vals[-1], y_vals[-1], a, b)
        x = normalize(x, -10, 10)  # Adjust the range as needed
        y = normalize(y, -10, 10)  # Adjust the range as needed
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

# Initial conditions and parameters
a, b = 0.9, 0.4
x0_1, y0_1 = 0, 0

iterations = 1000

x1_vals, y1_vals = generate_trajectory(x0_1, y0_1, a, b, iterations)



# Save trajectories into text files for ploting with gnuplot
# Change the names of the files for the questions from A to D.
with open('question_4c.txt', 'w') as file:
    for x, y in zip(x1_vals, y1_vals):
        file.write(f"{x} {y}\n")



