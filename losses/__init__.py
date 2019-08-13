from collections import namedtuple
import numpy as np
from functools import partial


_LossFunction = namedtuple('LossFunction', ['func', 'grad'])


# We are using a helper class to be able to add a docstring properly
class LossFunction(_LossFunction):
    """
    Named tuple representing a loss function.
    Contains the function (as a lambda) and its gradient (as a lambda).
    """
    pass


def _gaussian_mle_func(z):
    """
    Gaussian MLE loss function.
    :param z: Point of evaluation.
    :return: Loss value at z.
    """
    return z


def _gaussian_mle_grad(z):
    """
    Gaussian MLE loss gradient.
    :param z: Point of evaluation.
    :return: Gradient value at z.
    """
    return 1


# Gaussian MLE loss
gaussian_mle = LossFunction(
    func=_gaussian_mle_func,
    grad=_gaussian_mle_grad
)


def _tylers_estimator_func(z, d):
    """
    Tyler's estimator loss function.
    :param z: Point of evaluation.
    :param d: Dimension (number of features).
    :return: Loss value at z.
    """
    return d * np.log(z)


def _tylers_estimator_grad(z, d):
    """
    Tyler's estimator loss gradient.
    :param z: Point of evaluation.
    :param d: Dimension (number of features).
    :return: Gradient value at z.
    """
    return d / z


def tylers_estimator(d):
    """
    Returns a Tyler's estimator loss with dimension d.
    :param d: Dimension (number of features).
    :return: Tyler's loss function for dimension d.
    """
    #g = lambda z: d * np.log(z)
    #grad = lambda z: d / z
    #return LossFunction(g, grad)
    return LossFunction(
        partial(_tylers_estimator_func, d=d),
        partial(_tylers_estimator_grad, d=d)
    )


def _generalized_gaussian_func(z, beta, m):
    """
    Generalized Gaussian loss function.
    :param z: Point of evaluation.
    :param beta: Shape parameter.
    :param m: Scaling of scatter matrix.
    :return: Loss value at z.
    """
    return np.float_power(z, beta)


def _generalized_gaussian_grad(z, beta, m):
    """
    Generalized Gaussian loss gradient.
    :param z: Point of evaluation.
    :param beta: Shape parameter.
    :param m: Scaling of scatter matrix.
    :return: Gradient value at z.
    """
    return beta * (np.float_power(z, (beta - 1)))


def generalized_gaussian(beta, m):
    """
    Returns a Generalized Gaussian loss object.
    :param beta: Shape parameter.
    :param m: Scaling of scatter matrix.
    :return: Generalized Gaussian loss function for the given parameters.
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
    """
    Multivariate T loss function.
    :param z: Point of evaluation.
    :param nu: Degrees of freedom.
    :return: Loss value at z.
    """
    return (nu + d) * np.log(1 + z / nu)


def _multivariate_t_grad(z, d, nu):
    """
    Multivariate T loss gradient.
    :param z: Point of evaluation.
    :param nu: Degrees of freedom.
    :return: Gradient value at z.
    """
    return ((nu + d) / nu) * (1. / (1 + z / nu))


def multivariate_t(d, nu):
    """
    Returns a Multivariate T loss object.
    :param beta: Shape parameter.
    :param nu: Degrees of freedom.
    :return: Multivariate T loss function for the given nu.
    """
    #g = lambda z: (nu + d) * np.log(1 + z / nu)
    #grad = lambda z: ((nu + d) / nu) * (1. / (1 + z / nu))
    #return LossFunction(g, grad)
    return LossFunction(
        partial(_multivariate_t_func, d=d, nu=nu),
        partial(_multivariate_t_grad, d=d, nu=nu)
    )
