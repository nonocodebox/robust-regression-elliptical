from util import Nameable, PlotAdditionalParameters


class JointEstimator(Nameable, PlotAdditionalParameters):
    """
    A Joint inverse covariance estimator.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure.
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        raise NotImplementedError('This method must be implemented in a derived class')
