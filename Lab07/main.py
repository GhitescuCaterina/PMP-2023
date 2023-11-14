import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# a. setul de date
df = pd.read_csv('auto-mpg.csv')

# vizualizare preliminară a datelor
print(df.head())

# grafic
plt.scatter(df['CP'], df['mpg'])
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.title('Relația dintre CP și mpg')
plt.show()

# b. modelul în pymc
with pm.Model() as linear_model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    # modelul liniar
    mu = alpha + beta * df['CP']

    # likelihood
    likelihood = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])

# c. dreapta de regresie
with linear_model:
    trace = pm.sample(2000, tune=1000)

# rezultate
pm.traceplot(trace)
plt.show()

# d. regiunea 95%HDI pe grafic
az.plot_hdi(df['CP'], trace['mpg'])
plt.scatter(df['CP'], df['mpg'])
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.title('Dreapta de regresie cu HDI')
plt.show()
