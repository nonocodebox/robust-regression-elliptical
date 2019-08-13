import numpy as np
from ..joint import JointEstimator
import util


class SampleCovarianceJointEstimator(JointEstimator):
    """
    Inverse sample covariance estimator.
    """

    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Sample Covariance'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        return np.linalg.pinv(util.sample_covariance(X))
