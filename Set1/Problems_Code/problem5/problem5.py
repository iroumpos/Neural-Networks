import matplotlib.pyplot as plt
import numpy as np


def logsig(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x / (1 + np.exp(-x))


p = np.linspace(-2,2,1000)
w11 = -2
w12 = -1
b1 = -0.5
b12 = -0.75
w21 = 2
w22 = 1
b = 0.5


n_1 = w11*p + b1
n_2 = w12*p + b12
a12_log = logsig(n_1)
a12_swish = swish(n_1)
a21_log = logsig(n_2)
a21_swish = swish(n_2)

n2_log = a12_log*w21 + a21_log*w22 + b
n2_swish = a12_swish*w21 + a21_swish*w22 + b


plt.figure(figsize=(8,6))
plt.plot(p,a12_log, color = 'blue', label = 'a1')
plt.plot(p, a21_log, color = 'red', label = 'a2')
plt.plot(p, n2_log, color = 'magenta', label = 'a')
plt.xlabel('p')
plt.title('LogSig')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(p,a12_swish, color = 'blue', label = 'a1')
plt.plot(p, a21_swish, color = 'red', label = 'a2')
plt.plot(p, n2_swish, color = 'magenta', label = 'a')
plt.xlabel('p')
plt.title('Swish')
plt.legend()
plt.show()
