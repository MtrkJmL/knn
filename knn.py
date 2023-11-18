import numpy as np


def knn(samples, k):
    
    def euclidean(point, data):
        return np.sqrt(np.sum((point-data)**2, axis=0), dtype=float)

    pos = np.arange(-5, 5.0, 0.1)
    N = len(samples)
    estDensity = np.ndarray(shape=(N, 2), dtype=float)
    for j in range(len(pos)):
        dist=np.zeros(N)
        for i in range(N):
            dist[i] = euclidean(pos[j],samples[i])
        sorted = np.sort(dist)
        max_d = sorted[k-1]
        estDensity[j][1] = k/(N*2*max_d) #source=https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
        estDensity[j][0] = pos[j]
    return estDensity
