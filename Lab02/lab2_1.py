import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
m1 = stats.expon(0, 1 / 4).rvs(10000)
m2 = stats.expon(0, 1 / 6).rvs(10000)

p1 = 0.4
p2 = 0.6
x = p1 * m1 + p2 * m2

mean = np.mean(x)
deviation = np.std(x)
print("Average service time:", mean)
print("Deviation", deviation)

az.plot_posterior({'x': x})
plt.show()
