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


