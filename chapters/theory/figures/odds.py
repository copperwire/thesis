import matplotlib.pyplot as plt
import numpy as np


def odds(x: np.array): return x/(1-x)


p = np.linspace(1e-6, 0.99, 100)
odds(0.5)
#plt.plot(p, odds(p))
plt.plot(p, np.log(odds(p)))
plt.show()
