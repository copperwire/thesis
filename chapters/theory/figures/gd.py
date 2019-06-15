import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def f(x): return x**2


def df(x): return 2*x


etas = [7e-2, 3e-1, 1.1]
steps = [5, 5, 5]
names = ["low", "good", "large"]


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

for i, eta in enumerate(etas):
    x = [5, ]
    for k in range(steps[i]):
        x.append(x[k] - eta * df(x[k]))

    ex = max(x) + 3
    x_vals = np.linspace(-ex, ex, 100)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, c=colors[0], lw=2)
    plt.scatter(x, [f(j) for j in x], c=colors[5], s=100)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel(r"$\theta_1$", size=20)
    for j in range(len(x)-1):
        x1 = x[j]
        y1 = f(x1)

        x2 = x[j+1]
        y2 = f(x2)
        plt.annotate(
            "",
            xy=(x2, y2),
            xycoords="data",
            xytext=(x1, y1),
            textcoords="data",
            arrowprops=dict(
                width=0.9,
                headwidth=4,
                headlength=10,
                connectionstyle="arc3",
                color=colors[2],
            )
        )
    plt.savefig("gd_"+names[i]+".pdf")
    plt.clf()
    plt.cla()
