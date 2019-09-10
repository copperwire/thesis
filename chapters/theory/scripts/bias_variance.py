import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from estimator import estimator


class parallel_wrapper:
    def __init__(self, degrees):
        self.degrees = degrees
        self.biases = np.zeros(len(self.degrees))
        self.variances = np.zeros(len(self.degrees))
        self.e_outs = np.zeros(len(self.degrees))

    def __call__(self, est_inst, i, trials, trial_points):
        degree = self.degrees[i]
        bias, var, e_out = compute_bias_variance(
            est_inst, degree, trials, trial_points)
        self.biases[i] = bias
        self.variances[i] = var
        self.e_outs[i] = e_out


def compute_bias_variance(est_inst, degree, trials, trial_points):
    polynomials = est_inst.make_polynomials(degree, trials, trial_points)
    #polynomials = est_inst.make_neural_nets(degree, trials, trial_points)
    # evaluates the estimator trained on different training sets on 
    # the one test set making a (estimators, n_test) matrix
    test_eval = np.array([p(est_inst.test_set[0])
                          for p in polynomials])
    # Compute the expectation over the estimators generated from different
    # training sets
    expect_eval = test_eval.mean(0)
    # Evaluates the bias squared as eq. 7 in Mehta et. al 2019
    bias_sq = np.square(est_inst.test_set[1]- expect_eval).sum()
    # Evaluates the variance as eq. 8 in Mehta et. al 2019
    variance = ((np.square(test_eval - expect_eval)).mean(0)).sum()
    e_out = bias_sq + variance + len(est_inst.test_x)*est_inst.noise_params[1]**2
    return bias_sq, variance, e_out


#true_poly = np.poly1d([3.2e-2, 0.02, 0.8, 1.2, 2])
#true_poly = lambda x: np.exp(-(x-3)**2) + np.exp(-(x +1)**2) * 2* np.sin(x)
true_poly = lambda x: np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) 
trial_order = np.arange(0, 14)
est_inst = estimator(true_poly)
pw = parallel_wrapper(trial_order)
Parallel(n_jobs=10, require="sharedmem")(delayed(pw)(est_inst, i, 1000, 125)
                                        for i in range(len(trial_order)))

fig, ax = plt.subplots(figsize=(6, 5))
#ax2 = ax.twinx()
#ax3 = ax.twinx()
axs = [ax, ax, ax]
labels = [r"Bias$^2$", "Variance", r"$E_{out}$"]
cm = matplotlib.cm.get_cmap("magma")
colors = [cm(0.3), cm(0.6), cm(0.85)]
outcomes = [pw.biases, pw.variances, pw.e_outs]
lines = []
for i, l in enumerate(labels):
    ln = axs[i].plot(trial_order, outcomes[i], label=labels[i], c=colors[i], lw=2)
    lines += ln
    axs[i].get_yaxis().set_ticks([])
    # axs[i].axes.get_yaxis().set_visible(False)

ax.set_ylim((0, max(pw.biases+pw.variances)+2))
labels = [l.get_label() for l in lines]
axs[0].legend(lines, labels, loc="best")
axs[0].set_xlabel("Model complexity")
axs[0].set_ylabel("Error")
plt.savefig("../figures/bias_var_degree.png")
plt.savefig("../figures/bias_var_degree.pdf")
