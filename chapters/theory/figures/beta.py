import matplotlib
import matplotlib.pyplot as plt
import numpy as np


inferno = matplotlib.cm.get_cmap('inferno')
viridis = matplotlib.cm.get_cmap('viridis')

colors = [
    viridis(0.1),
    viridis(0.3),
    viridis(0.5),
    viridis(0.7),
    inferno(0.5),
    inferno(0.6),
    inferno(0.7),
    inferno(0.9)
]

power = np.arange(1, 40)
betas = [0.95, 0.9, 0.5]

for beta in betas:
    plt.plot(power, beta**power, lw=2, c=inferno(beta-0.3),
             label=r"$\beta = {}$".format(beta))
    plt.xlabel("n")
    plt.yticks([])

plt.legend()
plt.savefig("beta_decay.pdf")
