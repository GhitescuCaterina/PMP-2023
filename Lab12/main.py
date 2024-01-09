import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# sectiunea 1: grid computing cu diferite distributii a priori

def compute_posterior(grid, prior, likelihood):
    unnormalized_posterior = likelihood * prior
    posterior = unnormalized_posterior / unnormalized_posterior.sum()
    return posterior


grid_points = 100
grid = np.linspace(0, 1, grid_points)
np.random.seed(0)
observed_data = np.random.binomial(n=1, p=0.8, size=50)
likelihood = np.array(
    [np.prod(grid_point ** observed_data * (1 - grid_point) ** (1 - observed_data)) for grid_point in grid])

priors = {
    "Binary Prior": (grid <= 0.5).astype(int),
    "Absolute Difference Prior": abs(grid - 0.5)
}

plt.figure(figsize=(12, 8))
for i, (prior_name, prior) in enumerate(priors.items()):
    posterior = compute_posterior(grid, prior, likelihood)
    plt.subplot(2, 1, i + 1)
    plt.plot(grid, posterior, label=f"Posterior with {prior_name}")
    plt.xlabel("Parameter Value")
    plt.ylabel("Probability")
    plt.title(f"Posterior Distribution using {prior_name}")
    plt.legend()
plt.tight_layout()
plt.show()


# sectiunea 2: estimarea lui π și a erorii

def estimate_pi_and_error(N, num_trials=100):
    errors = []
    for _ in range(num_trials):
        x, y = np.random.uniform(-1, 1, size=(2, N))
        inside = (x ** 2 + y ** 2) <= 1
        pi_estimate = inside.sum() * 4 / N
        error = abs((pi_estimate - np.pi) / np.pi) * 100
        errors.append(error)
    return np.mean(errors), np.std(errors)


Ns = [100, 1000, 10000]
num_trials = 100
mean_errors = []
std_errors = []

for N in Ns:
    mean_error, std_error = estimate_pi_and_error(N, num_trials)
    mean_errors.append(mean_error)
    std_errors.append(std_error)

plt.figure(figsize=(10, 6))
plt.errorbar(Ns, mean_errors, yerr=std_errors, fmt='-o')
plt.xlabel('Number of Points (N)')
plt.ylabel('Error in Estimation of π (%)')
plt.title('Error in π Estimation as a Function of Number of Points')
plt.xscale('log')
plt.grid(True)
plt.show()

print("erori medii:", mean_errors)
print("deviatia standard a erorilor:", std_errors)


# sectiunea 3: metoda metropolis si comparatia cu grid computing pentru o distributie beta

def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        if new_x > 0 and new_x < 1:
            new_prob = func.pdf(new_x)
            acceptance = new_prob / old_prob
            if acceptance >= np.random.random():
                trace[i] = new_x
                old_x = new_x
                old_prob = new_prob
            else:
                trace[i] = old_x
        else:
            trace[i] = old_x
    return trace


def grid_computing_beta(a, b, grid_points=100):
    grid = np.linspace(0, 1, grid_points)
    prior = stats.beta(a, b).pdf(grid)
    posterior = prior / prior.sum()
    return grid, posterior


func = stats.beta(2, 5)
trace = metropolis(func=func)
x = np.linspace(0.01, .99, 100)
y = func.pdf(x)

grid, posterior = grid_computing_beta(a=2, b=5)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'C1-', lw=3, label='True Beta Distribution')
sns.histplot(trace[trace > 0], bins=25, kde=False, stat="density", label='Metropolis Estimated Distribution',
             color='C2')
plt.plot(grid, posterior, 'C0--', lw=3, label='Grid Computing Distribution')
plt.xlim(0, 1)
plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.yticks([])
plt.legend()
plt.show()
