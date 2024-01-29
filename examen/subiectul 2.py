import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import geom

# 1. modificare codului de la pagina 7, cursul 11

# definim functia care calculeaza distributia a posteriori folosind o abordare de tip grila
def posterior_grid_geometric(grid_points=100, first_head=5):
    # cream o grila de valori pentru theta  intre 0.01 si 0.99 ca sa evit valorile extreme
    # calculam verosimilitatea folosind distributia geometrica, cu 'first_head' ca numarul de incercari pana la
    # prima reusita
    # calculam distributia a posteriori prin inmultirea verosimilitatii cu priorul si normalizarea rezultatului

    grid = np.linspace(0.01, 0.99, grid_points)
    prior = np.repeat(1 / grid_points, grid_points)
    likelihood = geom.pmf(first_head, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    # Verificarea daca toate valorile din distributia a posteriori sunt zero
    if np.all(posterior == 0):
        print("toate valorile din posterior sunt zero, trebuiesc ajustati parametrii.")
    else:
        # daca sunt valori nenule, functia returneaza grila de valori theta si distributia a posteriori corespunzatoare
        return grid, posterior


# setarea numarului de puncte din grila si a numarului de aruncari pana la obtinerea primei steme
grid_points = 10  # numarul depuncte din grila poate fi setat la oricat
first_head = 5  # la fel si aici, putem seta cand sa fie prima aparitie a stemei

grid, posterior = posterior_grid_geometric(grid_points, first_head)

# 2. theta maxim

max_posterior_idx = np.argmax(posterior)
theta_map = grid[max_posterior_idx]

print("Indexul valorii maxime din a posteriori: ", max_posterior_idx)
print("Valoarea maxima a lui theta: ", theta_map)

# aici cream si afisam graficul pentru distributia a posteriori
plt.plot(grid, posterior, 'o-', label='Posterior')
plt.axvline(theta_map, color='red', linestyle='--', label=f'Theta MAP = {theta_map:.4f}')
plt.title('Posterior distribution after first head on n-th toss')  # prima stema poate fi la a cata aruncare vrem noi
plt.xlabel('θ')
plt.ylabel('Probability')
plt.legend()
plt.show()
