import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cm = matplotlib.cm.get_cmap('magma')

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return sigmoid(x)*(1-sigmoid(x))

def tanh(x): return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def dtanh(x): return 1 - tanh(x)**2

def relu(x):
    mask = x < 0
    x[mask] = 0
    return x
def drelu(x):
    mask = x > 0
    return mask.astype(np.float32)

def lrelu(x, a=0.4):
    mask = x < 0
    x[mask] = x[mask]*a
    return x

def dlrelu(x, a=0.4):
    mask = x > 0
    mask = mask.astype(np.float32)
    mask[mask == 0] = a
    #print(mask)
    return mask

sigmoids = [
    sigmoid,
    tanh
]
dsigmoids = [
    dsigmoid,
    dtanh
]
elus = [
    relu,
    lrelu
]
delus = [
    drelu,
    dlrelu,
]
to_plot = "sigmoids"

if to_plot == "sigmoids":
    activations = [sigmoids,]
    dactivations = [dsigmoids,]
    names = [
    r"$\sigma (wx)$",
    r"$\tanh (wx)$"
    ]
else:
    activations = [elus,]
    dactivations = [delus,]
    names = [
    r"ReLU $(wx)$",
    r"LReLU $(wx)$"
    ]

x = np.linspace(-5, 5, 100)
weights = np.linspace(0.1, 1, 10)
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(8, 6))
for i in range(2):
    for w in weights:
        for j in range(len(activations[0])):
            ax[i][0].plot(x, activations[0][i](w*x), c=cm(w), alpha=0.6)
            ax[i][0].set_ylabel(names[i])
        for j in range(len(dactivations[0])):
            ax[i][1].plot(x, w*dactivations[0][i](w*x), c=cm(w), alpha=0.6)
            ax[i][j].set_xlabel("x")
            ax[i][1].set_ylabel(r"$d \,$"+names[i])
        #ax[0][0].set_ylim((-1.1, 1.1))
        #ax[1][0].set_xlim((-3, 5))

plt.subplots_adjust(hspace=0.4, wspace=0.3)
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(cmap=cm),
     ax=ax,
     orientation="horizontal",
     )
cbar.ax.set_xlabel("w")
#plt.tight_layout()
plt.savefig("../figures/activations"+to_plot+".png")
plt.savefig("../figures/activations"+to_plot+".pdf")

