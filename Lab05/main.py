import numpy as np
import pandas as pd
import pymc3 as pm

data = pd.read_csv('trafic.csv')
trafic = data['nr. masini'].values

with pm.Model() as model:
    lambda_ = pm.Gamma('lambda_', alpha=1, beta=0.1)

    trafic_observed = pm.Poisson('trafic_observed', mu=lambda_, observed=trafic)

    trace = pm.sample(10000, tune=1000, cores=1)

pm.summary(trace)

pm.plot_posterior(trace)

lambda_values = trace['lambda_']
top_intervals = np.argsort(lambda_values)[-5:]
top_intervals_values = [trafic[i] for i in top_intervals]
top_intervals_times = [f'Ora {i // 60 + 4}:{i % 60}' for i in top_intervals]

print(f"Cele mai probabile 5 intervale de timp:")
for i in range(5):
    print(f"Intervalul {top_intervals_times[i]} - Trafic: {top_intervals_values[i]}")

for i in top_intervals:
    print(
        f"Intervalul {top_intervals_times[i]}: Media = {np.mean(lambda_values)}, Deviația Standard = {np.std(lambda_values)}")
