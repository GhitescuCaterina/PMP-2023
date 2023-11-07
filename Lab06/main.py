# import pymc3
# import theano
#
# print(pymc3.__version__)
# print(theano.__version__)

# Un magazin este vizitat de n clienţi într-o anumită zi. Numărul Y de clienţi care cumpără un anumit produs
# e distribuit Binomial(n, θ), unde θ este probabilitatea ca un client să cumpere acel produs. Să presupunem că
# îl cunoaştem pe θ şi că distribuţia a priori pentru n este Poisson(10).
# Folosiţi PyMC pentru a calcula distribuţia a posteriori pentru n pentru toate combinaţiile de
# Y ∈ {0, 5, 10} şi θ ∈ {0.2, 0.5}. Folosiţi az.plot_posterior pentru a vizualiza toate rezultatele
# (ideal, într-o singură fereastră).

import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

# Y = 0, 5, 10
# theta = 0.2, 0.5
# n = Poisson(10)

# luam fiecare caz in parte

# Y = 0, theta = 0.2
n = 10000
Y = 0
theta = 0.2
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


# Y = 0, theta = 0.5
n = 10000
Y = 0
theta = 0.5
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


# Y = 5, theta = 0.2
n = 10000
Y = 5
theta = 0.2
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


# Y = 5, theta = 0.5
n = 10000
Y = 5
theta = 0.5
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


# Y = 10, theta = 0.2
n = 10000
Y = 10
theta = 0.2
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()


# Y = 10, theta = 0.5
n = 10000
Y = 10
theta = 0.5
data = np.random.binomial(n, theta, size=1)
print(data)
with pm.Model() as model:
    n = pm.Poisson('n', mu=10)
    y = pm.Binomial('y', n=n, p=theta, observed=data)
    trace = pm.sample(10000, tune=1000, cores=1)
    az.plot_posterior(trace)
    plt.show()
