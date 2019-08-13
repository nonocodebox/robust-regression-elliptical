import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor as SKHuberRegressor
from ..conditional import ConditionalRegressorBase
from ..joint import JointRegressorBase
import util


class DenseConditionalRegressorBase(JointRegressorBase, ConditionalRegressorBase):
    def __init__(self, dx=None, dy=None, **kwargs):
        super().__init__(**kwargs)
        self.dx = dx
        self.dy = dy

    def regress_joint(self, X, E, T):
        if self.dx is None or self.dy is None:
            raise Exception('dx or dy not set')

        if self.dx + self.dy != X.shape[0]:
            raise Exception('Dimension mismatch')

        _, E_yx, E_yy = util.split_edges(E, self.dx, self.dy)

        return self.regress_conditional(X[:self.dx], X[self.dx:], E_yx, E_yy, T)


class LinearRegressor(DenseConditionalRegressorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_name(self):
        return 'Linear'

    def regress_conditional(self, X, Y, E_yx, E_yy, T):
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X.T, Y.T)

        return lr.coef_


class HuberRegressor(DenseConditionalRegressorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_name(self):
        return 'Huber'

    def regress_conditional(self, X, Y, E_yx, E_yy, T):
        dx = X.shape[0]
        dy = Y.shape[0]
        coef = np.zeros((dy, dx))

        for i in range(dy):
            huber = SKHuberRegressor(fit_intercept=False)
            huber.fit(X.T, Y[i, :].T)
            coef[i, :] = huber.coef_

        return coef
