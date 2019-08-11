from ..joint import JointEstimator
import numpy as np
import util


class TylerJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using Tyler's algorithm.
    TODO: Cite paper
    """
    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Tyler'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure.
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        EPS = 1e-7
        p, N = X.shape

        if K_0 is not None:
            Q = np.linalg.pinv(K_0)
        else:
            # Initialize with PD matrix.
            Q = np.eye(p)
            # Q = np.random.randn(p, N)
            # Q = util.sample_covariance(Q)

        prevQ = Q

        for i in range(T):
            d = np.sum(X * np.dot(np.linalg.pinv(Q), X), axis=0)
            Q = 1.0 * p / N * np.dot(X,np.dot(np.diag(1./d), X.T))

            if np.linalg.norm(prevQ - Q, 'fro') <= EPS:
                break

            prevQ = Q

        return np.linalg.pinv(Q)
