import numpy as np
from util import Nameable, PlotAdditionalParameters


class ConditionalRegressorBase(Nameable, PlotAdditionalParameters):
    """
    Base class for conditional regressors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def regress_conditional(self, X, Y, E_yx, E_yy, T):
        """
        Returns regression coefficients.
        :param X: Input features matrix of size (number of features, number of samples).
        :param Y: Targets matrix of size (number of targets, number of samples).
        :param E_yx: Prior targets-features structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param E_yy: Prior targets structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :return: Regression coefficient matrix of size (number of targets, number of features).
        """
        raise Exception('Method must be overridden in a derived class')


class ConditionalRegressor(ConditionalRegressorBase):
    """
    A regressor based on conditional covariance estimation.
    """

    def __init__(self, conditional_estimator, **kwargs):
        """
        Initializes the regressor.
        :param conditional_estimator: The conditional estimator to use for estimating regression coefficients.
        """
        super().__init__(**kwargs)
        self.estimator = conditional_estimator

    def default_name(self):
        return self.estimator.name()

    def regress_conditional(self, X, Y, E_yx, E_yy, T):
        """
        Returns regression coefficients.
        :param X: Input features matrix of size (number of features, number of samples).
        :param Y: Targets matrix of size (number of targets, number of samples).
        :param E_yx: Prior targets-features structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param E_yy: Prior targets structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :return: Regression coefficient matrix of size (number of targets, number of features).
        """
        K_yy, K_yx = self.estimator.estimate_conditional(X, Y, E_yx, E_yy, T)
        return -np.linalg.inv(K_yy) @ K_yx
