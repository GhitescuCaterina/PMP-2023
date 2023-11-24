import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# parametri generarea datelor
mu = 10
sigma = 2
num_samples = 200  # numărul de timpi medii de așteptare

# generarea a 200 de timpi medii de asteptare folosind distributia normala
timpi_medii_asteptare = np.random.normal(loc=mu, scale=sigma, size=num_samples)

# calculul mediei teoretice
media_teoretica = np.mean(timpi_medii_asteptare)
print(f"Media teoretică: {media_teoretica}")

# histrograma
plt.hist(timpi_medii_asteptare, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histograma timpilor medii de așteptare')
plt.xlabel('Timp mediu de așteptare')
plt.ylabel('Frecvență relativă')

# linia pt media teoretica
plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='Media teoretică')
plt.legend()

# afisarea histogramei
plt.show()

with pm.Model() as model:
    sigma = pm.Beta('sigma', alpha=1., beta=1.)
    y = pm.Bernoulli('y', p=sigma)
    timp_asteptare = pm.Normal("timp_asteptare", mu=y, sigma=sigma)
    observation = pm.Poisson("obs", mu=timp_asteptare, observed=media_teoretica)

with model:
    trace = pm.sample(1000, cores=1)
    az.plot_posterior(trace)
    plt.show()
