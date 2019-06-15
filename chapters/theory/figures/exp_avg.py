import numpy as np
import matplotlib.pyplot as plt

n_samp = 1000
x = np.linspace(-4, 4, n_samp)
y = x**2 + np.sin(4*x)*5 + np.random.normal(0, 1, size=n_samp)

y_smooth = np.zeros(n_samp)
prev = 0
beta = 0.95
patient = False
patience = 5
to_annot = []

for i in range(n_samp):
    y_smooth[i] = (prev*beta + (1 - beta)*y[i])
    prev = y_smooth[i]

    if y_smooth[i] > y_smooth[i-1]:
        if not patient:
            patient_i = i
        patient = True
    else:
        patient = False

    if patient and (i - patient_i) == patience:
        change = np.diff(y_smooth[patient_i:i])
        mean_change = change.mean()
        to_annot.append((x[i], y_smooth[i], mean_change))
        patient = False
        # print("----------")
        # print(change)
        # print(mean_change)


plt.plot(x, y, label="OG")
plt.plot(x, y_smooth, label="SMOOTH")

for t in to_annot:
    if t[2] > 0:
        marker = "r+"
    else:
        marker = "b-"
    plt.plot(t[0], t[1], marker)
plt.legend()
plt.show()
