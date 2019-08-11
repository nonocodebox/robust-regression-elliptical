from util import Nameable, PlotAdditionalParameters


class ConditionalEstimator(Nameable, PlotAdditionalParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def estimate_conditional(self, X, Y, Eyx, Eyy, T, Kyx_0=None, Kyy_0=None):
        raise NotImplementedError('This method must be implemented in a derived class')
