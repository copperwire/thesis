import matplotlib.pyplot as plt
import matplotlib
import numpy as np

cm  = matplotlib.cm.get_cmap('magma')
fig, axs = plt.subplots(ncols=2)
n_p = 30
train_i = int(np.ceil(n_p*0.65))
X = np.linspace(-2, 2, n_p)
a = 2
c = 0.8
d = 0.5
b = 1
y_1 = a*X + b + np.random.normal(0, 1, n_p)
y_2 = a*X + c*X**2 + d*X**3 + b + np.random.normal(0, 1, n_p)
y_a = [y_1, y_2]

for i in range(len(y_a)):
	y = y_a[i]
	ax = axs[i]
	deg = [1, 3, 7, 10]
	colors = cm(np.linspace(0.3, 0.8, len(deg)))
	for i, d in enumerate(deg):

		p = np.poly1d(np.polyfit(X[:train_i], y[:train_i], d))
		ax.plot(
			np.linspace(-2, 2, 1000), 
			p(np.linspace(-2, 2, 1000)),
			c=colors[i],
			alpha=0.5,
			label=r"Solution from $P^{{ {} }}$".format(d),
			linewidth=2,
			)


	ax.scatter(
		X[:train_i],
		y[:train_i],
		s=6,
		c="black",
		marker="^",
		label="train region"
		)

	ax.scatter(
		X[train_i:],
		y[train_i:],
		s=8,
		c="black",
		marker=".",
		label="test region"
		)
	ax.set_ylim((min(y)-1, max(y)+1))
	ax.set_xlabel("x", size=15)
axs[0].set_ylabel("y", size=15)	
ax.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.15),
          ncol=3, fancybox=True, shadow=True)

#plt.tight_layout()

plt.savefig("../figures/y_distr.png")
plt.close(fig)