from collections import namedtuple
import numpy as np
from functools import partial


LossFunction = namedtuple('LossFunction', ['func', 'grad'])


def _gaussian_mle_func(z):
    return z


def _gaussian_mle_grad(z):
    return 1


gaussian_mle = LossFunction(
    func=_gaussian_mle_func,
    grad=_gaussian_mle_grad
)


def _tylers_estimator_func(z, d):
    return d * np.log(z)


def _tylers_estimator_grad(z, d):
    return d / z


def tylers_estimator(d):
    """ Required params: d - number of features in input
    """
    #g = lambda z: d * np.log(z)
    #grad = lambda z: d / z
    #return LossFunction(g, grad)
    return LossFunction(
        partial(_tylers_estimator_func, d=d),
        partial(_tylers_estimator_grad, d=d)
    )


def _generalized_gaussian_func(z, beta, m):
    return np.float_power(z, beta)


def _generalized_gaussian_grad(z, beta, m):
    return beta * (np.float_power(z, (beta - 1)))


def generalized_gaussian(beta, m):
    """ Required params: beta - shape parameter, m - scaling of scatter matrix
    """
    #     g = lambda z: (z**params['beta'])/(2*(params['m']**params['beta']))
    #g = lambda z: (np.float_power(z, beta))
    #grad = lambda z: (beta * (np.float_power(z, (beta - 1))))
    #return LossFunction(g, grad)
    return LossFunction(
        partial(_generalized_gaussian_func, beta=beta, m=m),
        partial(_generalized_gaussian_grad, beta=beta, m=m)
    )


def _multivariate_t_func(z, d, nu):
    return (nu + d) * np.log(1 + z / nu)


def _multivariate_t_grad(z, d, nu):
    return ((nu + d) / nu) * (1. / (1 + z / nu))


def multivariate_t(d, nu):
    """ Required params: d - number of features, nu - degrees of freedom
    """
    #g = lambda z: (nu + d) * np.log(1 + z / nu)
    #grad = lambda z: ((nu + d) / nu) * (1. / (1 + z / nu))
    #return LossFunction(g, grad)
    return LossFunction(
        partial(_multivariate_t_func, d=d, nu=nu),
        partial(_multivariate_t_grad, d=d, nu=nu)
    )
