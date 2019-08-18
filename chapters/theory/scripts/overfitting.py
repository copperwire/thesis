import matplotlib.pyplot as plt
import matplotlib
import numpy as np

cm  = matplotlib.cm.get_cmap('magma')
fig, axs = plt.subplots(ncols=2, nrows=2)
n_p = 65
train_i = int(np.ceil(n_p*0.6))
X = np.linspace(-2, 4, n_p)
a = 2
c = 0.8
d = 0.5
b = 1
y_1 = a*X + b + np.random.normal(0, 1, n_p)
y_2 = a*X + c*X**2 + d*X**3 + b + np.random.normal(0, 2, n_p)
y_a = [y_1, y_2]


for i in range(len(y_a)):
	y = y_a[i]
	ax = axs[i]
	deg = [1, 3, 7, 10]
	colors = cm(np.linspace(0.3, 0.8, len(deg)))
	for k, d in enumerate(deg):
		p = np.poly1d(np.polyfit(X[:train_i], y[:train_i], d)) 
		ax[0].plot(
			X[:train_i], 
			p(X[:train_i]),
			c=colors[k],
			alpha=0.5,
			label=r"Solution from $P^{{ {} }}$".format(d),
			linewidth=2,
			)

		ax[1].plot(
			X[train_i:], 
			p(X[train_i:]),
			c=colors[k],
			alpha=0.5,
			label=r"Solution from $P^{{ {} }}$".format(d),
			linewidth=2,
			)
		ax[0].set_ylim((min(y[:train_i])-1, max(y[:train_i])+1))
		ax[1].set_ylim((min(y[train_i:])-1, max(y[train_i:])+1))

	ax[0].scatter(
		X[:train_i],
		y[:train_i],
		s=6,
		c="black",
		marker=".",
		label="Data"
		)

	ax[1].scatter(
		X[train_i:],
		y[train_i:],
		s=8,
		c="black",
		marker=".",
		label="Data"
		)

	if i == 0:
		ax[0].set_title("Training region")
		ax[1].set_title("Testing region")
	for a in ax:
		a.set_xlabel("x", size=15)

axs[0][0].set_ylabel("y", size=15)
axs[1][0].set_ylabel("y", size=15)		
lgnd = axs[0][0].legend(
	loc='upper center', bbox_to_anchor=(1, 1.8),
          ncol=3, fancybox=True, shadow=True)

#plt.tight_layout()

plt.savefig("../figures/y_distr.png", bbox_extra_artists=[lgnd, ], bbox_inches="tight")
plt.close(fig)