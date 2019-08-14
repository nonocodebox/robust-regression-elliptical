import numpy as np
import sklearn.datasets as datasets

"""
Generate a PSD covariance matrix of synthetic data with different structures.
"""

def generate_pd_symmetric_principal_diag(p, clique_size):
    """
    Generate PSD inverse covariance with tree structure (banded).
    :param p: The dimension.
    :param clique_size: Number of symmetric diagonals. 1 = 1 diagonal, 2 = 3 diagonals, 3 = 5 diagonals etc.
    :return The PSD array.
    """

    num_cliques = p - clique_size + 1
    D = np.zeros((p,p))

    for i in range(num_cliques):
        # Add transformed clique
        block_mat = np.random.normal(size=(clique_size, clique_size))
        block_mat = np.matmul(block_mat, block_mat.T)

        D[i:i+clique_size, i:i+clique_size] += block_mat

    return D


def generate_random_sparse_psd(p, zero_entry_chance=0.75):
    """
    Generate a random sparse PSD array.
    :param p: The dimension.
    :param zero_entry_chance: Zero-entry chance.
    :return: The PSD array.
    """
    return datasets.make_sparse_spd_matrix(p, alpha=zero_entry_chance)


def generate_3_banded_structure(p):
    """
    Generates a 3-banded structure.
    :param p: The dimension.
    :return: The structure - a list of tuples of edge indices (i, j).
    """
    E = []
    for i in range(p):
        if i < p-1:
            E.append((i,i+1))
        E.append((i,i))
        if i > 0:
            E.append((i,i-1))
    return E


def generate_inverse_covariance_structure(K, eps=1e-7):
    """
    Generates the structure list for a given inverse covariance array.
    :param K: The inverse covariance structure.
    :param eps: The tolerance for determining zero entries.
    :return: The structure - a list of tuples of edge indices (i, j).
    """
    p = K.shape[0]
    E = []
    for i in range(p):
        for j in range(p):
            if np.abs(K[i,j]) > eps:
                E.append((i,j))
    return E


def multivariate_generalized_gaussian(sigma, beta=1, dimension=1, size=1):
    """
    Sample variables from the multivariate generalized gaussian distribution.
    beta=1 gives normal distribution, beta=0.5 gives Laplace distribution.
    beta->0 converges to uniform distribution.
    :param sigma: The covariance matrix
    :param beta: The shape parameter
    :param dimension: The dimension
    :param size: The number of sampels to generate
    :return: The sampled values
    """

    u = np.random.standard_normal(size=dimension * size)
    u = np.reshape(u, (dimension, size))
    u /= np.linalg.norm(u, axis=0)

    tau = np.random.gamma(dimension / (2 * beta), 2, size=size) ** (1 / (2 * beta))

    return tau * (sigma @ u)
