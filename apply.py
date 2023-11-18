import numpy as np
from parameters import parameters
from knn import knn
from gauss1D import gauss1D
import matplotlib.pyplot as plt

h, k = parameters()

# Produce the random samples
samples = np.random.normal(0, 1, 100)

# Estimate the probability density using KNN
estDensity = knn(samples, k)

# Plot the distributions
plt.subplot(2, 1, 2)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KNN Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()
