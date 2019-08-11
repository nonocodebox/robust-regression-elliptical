import numpy as np
from util import Nameable, PlotAdditionalParameters


class ConditionalRegressorBase(Nameable, PlotAdditionalParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def regress_conditional(self, X, Y, Eyx, Eyy, T):
        raise Exception('Method must be overridden in a derived class')


class ConditionalRegressor(ConditionalRegressorBase):
    def __init__(self, conditional_estimator, **kwargs):
        super().__init__(**kwargs)
        self.estimator = conditional_estimator

    def default_name(self):
        return self.estimator.name()

    def regress_conditional(self, X, Y, Eyx, Eyy, T):
        Kyy, Kyx = self.estimator.estimate_conditional(X, Y, Eyx, Eyy, T)
        return -np.linalg.inv(Kyy) @ Kyx
