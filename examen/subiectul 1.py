import pandas as pd
import pymc as pm
import numpy as np

# 1. incarcati setul de date intr-un pandas data frame


housing = pd.read_csv('BostonHousing.csv')
valoarea_locuintelor = housing['medv'].values
nr_camere = housing['rm'].values
crima = housing['crim'].values
suprafata = housing['indus'].values

# print(valoarea_locuintelor, "\n", nr_camere, "\n", crima, "\n", suprafata)

housing.head()

# 2. definiti modelul in pymc folosind variabilele independente (rm, crim, indus) pentru a prezice variabila
# dependenta (medv)

