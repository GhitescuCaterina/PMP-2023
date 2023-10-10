import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

prob_stema = 0.3

num_aruncari = 10

rezultate = np.random.random_sample((100, num_aruncari, 2)) < prob_stema

num_ss = np.sum(np.all(rezultate, axis=2), axis=1)
num_sb = np.sum(rezultate[:, :, 0] & ~rezultate[:, :, 1], axis=1)
num_bs = np.sum(~rezultate[:, :, 0] & rezultate[:, :, 1], axis=1)
num_bb = np.sum(~rezultate, axis=2)

rezultate_posibile = ['ss', 'sb', 'bs', 'bb']
num_aparitii = [np.sum(num_ss), np.sum(num_sb), np.sum(num_bs), np.sum(num_bb)]

plt.bar(rezultate_posibile, num_aparitii)
plt.title('Distribuția Rezultatelor')
plt.xlabel('Rezultat')
plt.ylabel('Număr de apariții')
plt.show()
