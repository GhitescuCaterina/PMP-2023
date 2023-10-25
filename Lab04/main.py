import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

# exercitiul 1
model = pm.Model()

with model:
    alpha = 4
    trafic = pm.Poisson("T", alpha)
    plasare = pm.Normal("P", mu=2, sigma=0.5)
    pregatire = pm.Exponential("preg", 1/alpha)
    trace = pm.sample(20000)

dictionary = {
    'timp_pregatire': trace['preg'].tolist(),
    'timp_asteptare': trace['P'].tolist()
}
df = pd.DataFrame(dictionary)

# exercitiul 2
timpAsteptareSub15min = df[(df['timp_asteptare'] + df['timp_pregatire'] <= 15)]
size_sample = df.shape[0]
procentajAsteptareSub15min = timpAsteptareSub15min.shape[0] / size_sample
print("Procentajul timpului de asteptare sub 15 minute este:", procentajAsteptareSub15min)

# exercitiul 3
timpMediuAsteptare = df['timp_asteptare'].mean()
print("Timpul mediu de asteptare pentru a fi servit al unui client este:", timpMediuAsteptare)

az.plot_posterior({"Clienti pe ora": trace['T'], "Timp asteptare" : trace['P'], "Media pt comanda" : trace['preg']})
