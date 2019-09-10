import numpy as np
from sklearn.neural_network import MLPRegressor


class mlpwrapper(MLPRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, X):
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        return self.predict(X)

    def fit_1d(self, *args, **kwargs):
        args = list(args)
        if len(args[0].shape) == 1:
             args[0] = args[0].reshape((-1, 1))
        return self.fit(*args, **kwargs)

class estimator:
    """
    Estimator class that maintains the true values 
    generated from a groun-truth process and fits models to data 
    fro bias variance estimation for know processes. 
    """
    def __init__(
            self,
            true_poly,
            test_range=(2, 4),
            train_width=20,
            test_n=150,
            noise_params=(0, 0.1)
    ):
        """
        Initializes the class with a test set decomposed in 
        noise and true-process parts for bias-variance computation
        """
        self.true_poly = true_poly
        self.noise_params = noise_params
        self.test_x = np.linspace(
            test_range[0], test_range[1], test_n)
        self.f_test = self.true_poly(self.test_x)
        self.test_set = (self.test_x, self.f_test + np.random.normal(
            noise_params[0], noise_params[1], test_n))
        self.train_lim = (test_range[0]-train_width, test_range[0]+2)

    def _make_x(self, n_points, xlim):
        proto_x = np.linspace(xlim[0], xlim[1], 1e5)
        x =  np.setdiff1d(proto_x, self.test_x)
        ind = np.arange(0, len(x))
        if len(x) < n_points: 
            return np.zeros(1)
        x = x[np.random.choice(ind, n_points)]
        return x

    def make_data(self, n_points, xlim=(-2, 2)):
        n_x = 0
        while n_x != n_points:
            x = self._make_x(n_points, xlim)
            n_x = len(x)
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
            network = mlpwrapper([width, ]*1, alpha=0, max_iter=1000)
            network.fit_1d(x, y)
            networks.append(network)
        return networks
