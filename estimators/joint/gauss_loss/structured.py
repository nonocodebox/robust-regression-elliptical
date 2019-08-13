from ..joint import JointEstimator
import numpy as np
import util


class StructuredJointEstimator(JointEstimator):
    """
    Inverse covariance estimator for tree structure.
    The estimated sample covariance in this case has a closed form solution.
    """
    def __init__(self, clique_size, **kwargs):
        """
        Initialize the estimator.
        :param clique_size: Number of symmetric diagonals. 1 = 1 diagonal, 2 = 3 diagonals, 3 = 5 diagonals etc.
        """
        super().__init__(**kwargs)
        self.clique_size = clique_size

    def default_name(self):
        return 'Structured'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        """
        Implementation of closed form solution to the maximum likelihood under tree
        graph constraints.
        """
        # Input is cov, it the closed form we calculate ((K^-1)_c,c)^-1 = (Q_c,c)^-1,
        # Therefore, Q should be the input.
        return _transform_cliques(util.sample_covariance(X), self.clique_size, np.linalg.pinv)


def _transform_cliques(Q, clique_size, T):
    """
    Transforms every clique and intersection.
    :param Q: The input inverse covariance matrix.
    :param clique_size: The clique size.
    :param T: Transformation to apply.
    :return: The transformed matrix.
    """
    num_cliques = Q.shape[0] - clique_size + 1
    D = np.zeros(Q.shape)

    for i in range(num_cliques):
        # Add transformed clique
        D += _transform_clique(Q, clique_size, i, T)

        if i > 0:
            # Subtract transformed intersection
            D -= _transform_clique(Q, clique_size - 1, i, T)

    return D


def _transform_clique(Q, clique_size, i, T):
    """
    Transforms a single clique.
    :param Q: The input inverse covariance matrix.
    :param clique_size: The clique size.
    :param i: The starting row = column (symmetric).
    :param T: The transformation to apply.
    :return: The transformed matrix.
    """
    D = np.zeros(Q.shape)
    D[i:i+clique_size, i:i+clique_size] = T(Q[i:i+clique_size, i:i+clique_size])
    return D
