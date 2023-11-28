import pymc as pm
import pandas as pd
import numpy as np
import arviz as az


data = pd.read_csv('prices.csv')

price = data['Price']
processor_speed = data['Speed']
hard_drive_size = np.log(data['HardDrive'])


with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * processor_speed + beta2 * hard_drive_size

    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=price)

    trace = pm.sample(2000, tune=1000, cores=1)  # Adjustează numărul de eșantioane și tunare după nevoie

hdi_summary = az.summary(trace, hdi_prob=0.95)
hdi_beta1 = hdi_summary.loc['beta1', ['hdi_2.5%', 'hdi_97.5%']].values
hdi_beta2 = hdi_summary.loc['beta2', ['hdi_2.5%', 'hdi_97.5%']].values

print(f'Estimările HDI pentru beta1: {hdi_beta1}')
print(f'Estimările HDI pentru beta2: {hdi_beta2}')