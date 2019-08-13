from util import Nameable, PlotAdditionalParameters


class ConditionalEstimator(Nameable, PlotAdditionalParameters):
    """
    A Conditional inverse covariance estimator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def estimate_conditional(self, X, Y, E_yx, E_yy, T, K_yx_0=None, K_yy_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Input feature matrix of size (number of features, number of samples)
        :param Y: Target matrix of size (number of targets, number of samples)
        :param E_yx: Prior targets-features structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param E_yy: Prior targets structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_yx_0: Initial value for the estimated K_yx matrix.
        :param K_yy_0: Initial value for the estimated K_yy matrix.
        :return: Inverse covariance matrices (K_yx, K_yy)
                 K_yx is the targets-features inverse covariance matrix.
                 K_yy is the targets inverse covariance matrix.
        """
        raise NotImplementedError('This method must be implemented in a derived class')
