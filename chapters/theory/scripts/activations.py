import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cm = matplotlib.cm.get_cmap('magma')


def sigmoid(x): return 1/(1+np.exp(-x))


def relu(x):
    mask = x < 0
    x[mask] = 0
    return x


def lrelu(x, a=0.1):
    mask = x < a
    x[mask] = 0
    return x


functions = [
    sigmoid,
    np.tanh,
    relu,
    lrelu
]
x = np.linspace(-5, 5, 1e3)
for f in functions:
