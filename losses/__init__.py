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


def _tyler_func(z, d):
    """
    Tyler's loss function.
    :param z: Point of evaluation.
    :param d: Dimension (number of features).
    :return: Loss value at z.
    """
    return d * np.log(z)


def _tyler_grad(z, d):
    """
    Tyler's loss gradient.
    :param z: Point of evaluation.
    :param d: Dimension (number of features).
    :return: Gradient value at z.
    """
    return d / z


def tyler(d):
    """
    Returns a Tyler's loss with dimension d.
    :param d: Dimension (number of features).
    :return: Tyler's loss function for dimension d.
    """
    return LossFunction(
        partial(_tyler_func, d=d),
        partial(_tyler_grad, d=d)
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
    Returns a multivariate T loss object.
    :param beta: Shape parameter.
    :param nu: Degrees of freedom.
    :return: Multivariate T loss function for the given nu.
    """
    return LossFunction(
        partial(_multivariate_t_func, d=d, nu=nu),
        partial(_multivariate_t_grad, d=d, nu=nu)
    )
