import numpy as np
from util import Nameable, PlotAdditionalParameters


class JointRegressorBase(Nameable, PlotAdditionalParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def regress_joint(self, X, E, T):
        raise Exception('Method must be overridden in a derived class')


class JointRegressor(JointRegressorBase):
    def __init__(self, dx, dy, joint_estimator, **kwargs):
        super().__init__(**kwargs)

        self.dx = dx
        self.dy = dy
        self.estimator = joint_estimator

    def default_name(self):
        return self.estimator.name()

    def regress_joint(self, X, E, T):
        K = self.estimator.estimate_joint(X, E, T)
        Kyy = K[self.dx:, self.dx:]
        Kyx = K[self.dx:, :self.dx]
        return -np.linalg.inv(Kyy) @ Kyx
