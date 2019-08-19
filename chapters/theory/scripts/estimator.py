import numpy as np
from sklearn.neural_network import MLPRegressor


class mlpwrapper(MLPRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, X):
        return self.predict(X)


class estimator:
    def __init__(
            self,
            true_poly,
            test_range=(2, 4),
            train_width=20,
            test_n=150,
            noise_params=(0, 2)
    ):
        self.true_poly = true_poly
        self.noise_params = noise_params
        self.test_x = np.linspace(
            test_range[0], test_range[1], test_n)
        self.f_test = self.true_poly(self.test_x)
        self.test_set = (self.test_x, self.f_test + np.random.normal(
            noise_params[0], noise_params[1], test_n))
        self.train_lim = (test_range[0]-train_width, test_range[0]-0.1)

    def make_data(self, n_points, xlim=(-2, 2)):
        x = np.linspace(xlim[0], xlim[1], n_points)
        y = self.true_poly(x)
        y += np.random.normal(self.noise_params[0],
                              self.noise_params[1], n_points)
        return x, y

    def make_polynomials(self, degree, n, train_points):
        polynomials = []
        for i in range(n):
            x, y = self.make_data(train_points, self.train_lim)
            p = np.poly1d(np.polyfit(x, y, degree))
            polynomials.append(p)
        return polynomials

    def make_neural_nets(self, width, n, train_points):
        networks = []
        for i in range(n):
            x, y = self.make_data(train_points, self.train_lim)
            network = mlpwrapper([5, ]*width, alpha=0, max_iter=1000)
            x = x.reshape((-1, 1))
            network.fit(x, y)
            networks.append(network)
        return networks
