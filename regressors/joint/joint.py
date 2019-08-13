import numpy as np
from util import Nameable, PlotAdditionalParameters


class JointRegressorBase(Nameable, PlotAdditionalParameters):
    """
    Base class for joint regressors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def regress_joint(self, X, E, T):
        """
        Returns regression coefficients.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :return: Regression coefficient matrix of size (number of targets, number of features).
        """
        raise Exception('Method must be overridden in a derived class')


class JointRegressor(JointRegressorBase):
    """
    A regressor based on joint covariance estimation.
    """

    def __init__(self, dx, dy, joint_estimator, **kwargs):
        """
        Initializes the regressor.
        :param dx: Number of features.
        :param dy: Number of targets.
        :param joint_estimator: The joint estimator to use for estimating regression coefficients.
        """
        super().__init__(**kwargs)

        self.dx = dx
        self.dy = dy
        self.estimator = joint_estimator

    def default_name(self):
        return self.estimator.name()

    def regress_joint(self, X, E, T):
        """
        Returns regression coefficients.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :return: Regression coefficient matrix of size (number of targets, number of features).
        """
        K = self.estimator.estimate_joint(X, E, T)
        K_yy = K[self.dx:, self.dx:]
        K_yx = K[self.dx:, :self.dx]
        return -np.linalg.inv(K_yy) @ K_yx
