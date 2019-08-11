from ..joint import JointEstimator
from ..gauss_loss.structured import _transform_cliques
import numpy as np


class StructuredTylerJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using our algorithm which combines structured
    and Tyler's normalized sample covariance.
    """
    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Structured Tyler'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure.
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        p, n = X.shape

        if K_0 is None:
            Q = np.random.randn(p, n)
            Q = np.dot(Q, Q.T)/n
        else:
            Q = np.linalg.pinv(K_0)

        for i in range(T):
            d = np.sum(X*np.dot(np.linalg.pinv(Q), X), axis=0)
            structured_input = 1.0*p / n*np.dot(X, np.dot(np.diag(1./d), X.T))
            Q = np.linalg.pinv(_transform_cliques(structured_input, 2, np.linalg.pinv))

        return np.linalg.pinv(Q)
