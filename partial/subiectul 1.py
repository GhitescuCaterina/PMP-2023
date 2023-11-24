import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt


# vedem moneda masluita
def aruncare_moneda(n, moneda_masluita):
    if moneda_masluita:
        return sum(random.choices([0, 1], weights=[2 / 3, 1 / 3], k=n))
    else:
        return sum(random.choices([0, 1], k=n))


def simulare_joc():
    jucator_start = random.choice([0, 1])
    jucator_masluit = random.choice([True, False])

    steme_jucator_start = aruncare_moneda(1, False)
    steme_jucator_opus = aruncare_moneda(steme_jucator_start + 1, jucator_masluit)

    if jucator_start == 0:
        castigator = "P0" if steme_jucator_start >= steme_jucator_opus else "P1"
    else:
        castigator = "P1" if steme_jucator_opus > steme_jucator_start else "P0"

    return castigator, steme_jucator_start


def simulare_jocuri(numar_jocuri):
    rezultate_jocuri = {"P0": 0, "P1": 0}
    rezultate_prime_runde = {"0": 0, "1": 0}

    for _ in range(numar_jocuri):
        castigator, steme_prima_runda = simulare_joc()
        rezultate_jocuri[castigator] += 1
        rezultate_prime_runde[str(steme_prima_runda)] += 1

    procentaj_p0 = (rezultate_jocuri["P0"] / numar_jocuri) * 100
    procentaj_p1 = (rezultate_jocuri["P1"] / numar_jocuri) * 100

    probabilitate_fata_monedei_0 = (rezultate_prime_runde["0"] / numar_jocuri) * 100
    probabilitate_fata_monedei_1 = (rezultate_prime_runde["1"] / numar_jocuri) * 100

    return rezultate_jocuri, procentaj_p0, procentaj_p1, probabilitate_fata_monedei_0, probabilitate_fata_monedei_1


# simulam 20.000 de jocuri
numar_jocuri = 20000
rezultate, procentaj_p0, procentaj_p1, probabilitate_fata_monedei_0, probabilitate_fata_monedei_1 = simulare_jocuri(
    numar_jocuri)

# rezultatele
print(f"Rezultate după {numar_jocuri} de jocuri:")
print(f"P0 a câștigat de {rezultate['P0']} ori ({procentaj_p0:.2f}%)")
print(f"P1 a câștigat de {rezultate['P1']} ori ({procentaj_p1:.2f}%)")
print(f"Probabilitatea de a obține fața 0 (pajura) a monedei în prima rundă: {probabilitate_fata_monedei_0:.2f}%")
print(f"Probabilitatea de a obține fața 1 (stema) a monedei în prima rundă: {probabilitate_fata_monedei_1:.2f}%")


# reata Bayesiana
model = BayesianNetwork([('Jucator_Start', 'Steme_Prima_Runda'),
                       ('Moneda_Masluita', 'Steme_Prima_Runda'),
                       ('Steme_Prima_Runda', 'Castigator')])

# definim cpd-uri
cpd_js = TabularCPD(variable='Jucator_Start', variable_card=2, values=[[0.5], [0.5]])
cpd_mm = TabularCPD(variable='Moneda_Masluita', variable_card=2, values=[[0.67], [0.33]])

# P(Steme_Prima_Runda | Jucator_Start, Moneda_Masluita)
cpd_sr = TabularCPD(variable='Steme_Prima_Runda', variable_card=2,
                    values=[[0.5, 0.3, 0.5, 0.7],
                            [0.5, 0.7, 0.5, 0.3]],
                    evidence=['Jucator_Start', 'Moneda_Masluita'],
                    evidence_card=[2, 2])

# P(Castigator | Jucator_Start, Moneda_Masluita, Steme_Prima_Runda)
cpd_c = TabularCPD(variable='Castigator', variable_card=2,
                   values=[[0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7],  # P(Castigator | Jucator_Start=0, Moneda_Masluita=0, Steme_Prima_Runda=0), P(Castigator | Jucator_Start=1, Moneda_Masluita=0, Steme_Prima_Runda=0), ...
                           [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3]],  # P(Castigator | Jucator_Start=0, Moneda_Masluita=1, Steme_Prima_Runda=0), P(Castigator | Jucator_Start=1, Moneda_Masluita=1, Steme_Prima_Runda=0), ...
                   evidence=['Jucator_Start', 'Moneda_Masluita', 'Steme_Prima_Runda'],
                   evidence_card=[2, 2, 2])


# adaugam CPD-urile la reteaua Bayesiana
model.add_cpds(cpd_js, cpd_mm, cpd_sr)
model.add_cpds(cpd_c)


for node in model.nodes():
    print(f"Nodul: {node}          parintii: {model.get_parents(node)}")

# verificam daca e valida reteaua
assert model.check_model()

# inferam variabilele
infer = VariableElimination(model)
prob_castigator_stiind_steme_0 = infer.query(variables=['Castigator'], evidence={'Steme_Prima_Runda': 0})
print(prob_castigator_stiind_steme_0)

# incercam sa desenam reteaua bayesiana
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()


