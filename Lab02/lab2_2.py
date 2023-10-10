import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

m1 = stats.gamma(4, scale=1/3).rvs(10000)
m2 = stats.gamma(4, scale=1/2).rvs(10000)
m3 = stats.gamma(5, scale=1/2).rvs(10000)
m4 = stats.gamma(5, scale=1/3).rvs(10000)

latency = stats.expon(scale=4).rvs(10000)

server_indices = np.random.choice([0, 1, 2, 3], size=10000, p=[0.25, 0.25, 0.30, 0.20])
servers = np.array([m1, m2, m3, m4])

x = servers[server_indices, np.arange(10000)] + latency

mean = np.mean(x)
deviation = np.std(x)
print("Average service time:", mean)
print("Standard deviation:", deviation)

prob_gt_3ms = np.mean(x > 3)
print("Probability of service time > 3 milliseconds:", prob_gt_3ms)

plt.figure(figsize=(10, 6))
az.plot_posterior({'x': x}, kind='hist', round_to=2, color='skyblue')
plt.title('Density of the Total Service Time Distribution', fontsize=16)
plt.xlabel('Total Service Time (milliseconds)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
