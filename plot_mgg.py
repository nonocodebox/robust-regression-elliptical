import numpy as np
import sklearn.datasets as datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def multivariate_generalized_gaussian(sigma, beta=1, dimension=1, size=1):
    u = np.random.standard_normal(size=dimension * size)
    u = np.reshape(u, (dimension, size))
    u /= np.linalg.norm(u, axis=0)

    tau = np.random.gamma(dimension / (2 * beta), 2, size=size) ** (1 / (2 * beta))

    return tau * (sigma @ u)

beta = 1#0.5
p = 2

#Q = datasets.make_sparse_spd_matrix(p, alpha=0.6)
Q = np.eye(p)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 35000
v = multivariate_generalized_gaussian(Q, beta, p, N).T

H, ex, ey = np.histogram2d(v[:, 0], v[:, 1], bins=50, density=True)
nx, ny = H.shape
cx = (ex[:-1] + ex[1:]) / 2
cy = (ey[:-1] + ey[1:]) / 2

z = np.array([(cx[i], cy[j], H[i, j]) for i in range(nx) for j in range(ny)])

ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=0.2)
plt.show()