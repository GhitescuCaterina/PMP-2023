import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

def main():
    data = pd.read_csv("Admission.csv")

    gre_scores = data["GRE"].values
    gpa_scores = data["GPA"].values
    admission_status = data['Admission'].values

    with pm.Model() as logistic_model:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=10)

        pi = pm.math.invlogit(beta_0 + beta_1 * gre_scores + beta_2 * gpa_scores)
        admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed=admission_status)

    with logistic_model:
        trace = pm.sample(1000, tune=1000, cores=2)

    beta_0_samples = trace['beta_0']
    beta_1_samples = trace['beta_1']
    beta_2_samples = trace['beta_2']

    decision_boundary = np.median(pm.math.invlogit(beta_0_samples + beta_1_samples * gre_scores + beta_2_samples * gpa_scores))

    gre_values = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
    gpa_values = np.linspace(data['GPA'].min(), data['GPA'].max(), 100)
    x_grid = np.meshgrid(gre_values, gpa_values)
    p_grid = pm.math.sigmoid(np.median(beta_0_samples) + np.median(beta_1_samples) * x_grid[0] + np.median(beta_2_samples) * x_grid[1])
    decision_boundary_grid = np.median(p_grid > 0.5)

    hdi_grid = az.hdi(p_grid.eval(), hdi_prob=0.94)
    plt.scatter(data['GRE'], data['GPA'], c=[f'C{x}' for x in data['Admission']])
    plt.contour(x_grid[0], x_grid[1], decision_boundary_grid, levels=[0.5], colors='k')
    plt.contourf(x_grid[0], x_grid[1], hdi_grid, levels=[hdi_grid.min(), 0.5, hdi_grid.max()], colors=['b', 'r', 'b'], alpha=0.5)
    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.legend(['Decision boundary', '94% HDI'])
    plt.show()

    student1 = np.array([[550, 3.5]])
    p_student1 = pm.math.sigmoid(np.median(beta_0_samples) + np.median(beta_1_samples) * student1[0, 0] + np.median(beta_2_samples) * student1[0, 1]).eval()
    hdi_student1 = az.hdi(p_student1, hdi_prob=0.9)
    print(f"90% HDI for a student with GRE=550 and GPA=3.5: {hdi_student1[0]}, {hdi_student1[1]}")

    student2 = np.array([[500, 3.2]])
    p_student2 = pm.math.sigmoid(np.median(beta_0_samples) + np.median(beta_1_samples) * student2[0, 0] + np.median(beta_2_samples) * student2[0, 1]).eval()
    hdi_student2 = az.hdi(p_student2, hdi_prob=0.9)
    print(f"90% HDI for a student with GRE=500 and GPA=3.2: {hdi_student2[0]}, {hdi_student2[1]}")

if __name__ == "__main__":
    main()
