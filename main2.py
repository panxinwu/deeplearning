# Package imports
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_extra_datasets

# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
# ### END CODE HERE ###

# X, Y = datasets[dataset]
# print(datasets[dataset])
X = np.array([[56, 4, 2, 1, 0, 0, 0, 2, 5, 1, 261, 1, 999, 0, 3, 1.1, 93.994, -36.4, 4.857, 5191],
    [57, 8, 2, 4, 2, 0, 0, 2, 5, 1, 149, 1, 999, 0, 3, 1.1, 93.994, -36.4, 4.857, 5191],
    [37, 8, 2, 4, 0, 1, 0, 2, 5, 1, 226, 1, 999, 0, 3, 1.1, 93.994, -36.4, 4.857, 5191],
    [37, 2, 1, 3, 2, 1, 0, 2, 5, 3, 591, 1, 999, 0, 3, 1.1, 93.994, -36.4, 4.856, 5191]])
Y = np.array([0, 0, 0, 1])
# print(X)
# print(Y)
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "no_structure":
    Y = Y%2

# Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.scatter(X[0, :], X[1, :],c=Y.reshape(X[0,:].shape),  s=40, cmap = plt.cm.Spectral)
# plt.savefig('step5.png')
plt.show()