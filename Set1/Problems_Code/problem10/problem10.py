# Patterns for class A and class B
class_a = [(0, 0), (0, 1), (1, 0), (-1, -1)]
class_b = [(2.1, 0), (0, -2.5), (1.6, -1.6)]

# Write the patterns to a data file with numerical labels
with open('patterns.dat', 'w') as file:
    file.write("# x y class\n")
    for point in class_a:
        file.write(f"{point[0]} {point[1]} 0\n")
    for point in class_b:
        file.write(f"{point[0]} {point[1]} 1\n")

