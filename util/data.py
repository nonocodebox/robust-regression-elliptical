import numpy as np
import sklearn.datasets as datasets


'''
Generate a PSD covariance matrix of synthetic data with different structures.
'''


def generate_pd_symmetric_principal_diag(p, clique_size):
    '''
    Generate PSD inverse covariance with tree structure (banded).
    '''
    num_cliques = p - clique_size + 1
    D = np.zeros((p,p))

    for i in range(num_cliques):
        # Add transformed clique
        block_mat = np.random.normal(size=(clique_size, clique_size))
        block_mat = np.matmul(block_mat, block_mat.T)

        D[i:i+clique_size, i:i+clique_size] += block_mat
    return D


def generate_random_sparse_psd(p, zero_entry_chance=0.75):
    '''
    Generate random sparse PSD.
    '''
    return datasets.make_sparse_spd_matrix(p, alpha=zero_entry_chance)


def build_graph_3_daig_tree(p):
    E = []
    for i in range(p):
        if i < p-1:
            E.append((i,i+1))
        E.append((i,i))
        if i > 0:
            E.append((i,i-1))
    return E


def build_graph_of_inverse_cov(K):
    EPS = 1e-7
    p = K.shape[0]
    E = []
    for i in range(p):
        for j in range(p):
            if np.abs(K[i,j]) > EPS:
                E.append((i,j))
    return E


def multivariate_generalized_gaussian(sigma, beta=1, dimension=1, size=1):
    """
    Sample variables from the multivariate generalized gaussian distribution.
    beta=1 gives normal distribution, beta=0.5 gives Laplace distribution.
    beta->0 coverges to uniform distribution.
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
