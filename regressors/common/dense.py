import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor as SKHuberRegressor
from ..conditional import ConditionalRegressorBase
from ..joint import JointRegressorBase
import util


class DenseConditionalRegressorBase(JointRegressorBase, ConditionalRegressorBase):
    """
    Base class for dense regressors, which can be used both as a joint and a conditional regressor.
    """

    def __init__(self, dx=None, dy=None, **kwargs):
        """
        Initializes the regressor.
        :param dx: Number of features. Can be None when used as a conditional regressor.
        :param dy: Number of targets. Can be None when used as a conditional regressor.
        """
        super().__init__(**kwargs)
        self.dx = dx
        self.dy = dy

    def regress_joint(self, X, E, T):
        """
        Returns regression coefficients.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :return: Regression coefficient matrix of size (number of targets, number of features).
        """

        # Make sure we have both dimensions (features and targets)
        if self.dx is None or self.dy is None:
            raise Exception('dx or dy not set')

        # Make sure the dimensions add up
        if self.dx + self.dy != X.shape[0]:
            raise Exception('Dimension mismatch')

        # Split the edges
        _, E_yx, E_yy = util.split_edges(E, self.dx, self.dy)

        # Perform conditional regression
        return self.regress_conditional(X[:self.dx], X[self.dx:], E_yx, E_yy, T)


class LinearRegressor(DenseConditionalRegressorBase):
    """
    Linear regressor.
    Can be used both as a joint and a conditional regressor.
    """

    def __init__(self, **kwargs):
        """
        Initializes the regressor.
        :param dx: Number of features. Can be None when used as a conditional regressor.
        :param dy: Number of targets. Can be None when used as a conditional regressor.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Linear'

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
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X.T, Y.T)

        return lr.coef_


class HuberRegressor(DenseConditionalRegressorBase):
    """
    Huber regressor.
    Can be used both as a joint and a conditional regressor.
    """

    def __init__(self, **kwargs):
        """
        Initializes the regressor.
        :param dx: Number of features. Can be None when used as a conditional regressor.
        :param dy: Number of targets. Can be None when used as a conditional regressor.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Huber'

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
        dx = X.shape[0]
        dy = Y.shape[0]
        coef = np.zeros((dy, dx))

        for i in range(dy):
            huber = SKHuberRegressor(fit_intercept=False)
            huber.fit(X.T, Y[i, :].T)
            coef[i, :] = huber.coef_

        return coef
