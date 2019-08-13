from util import Nameable
import numpy as np


class ErrorMetric(Nameable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, estimator_index, estimator, N_index, N, i):
        raise NotImplementedError('Method must be overridden in a derived class')

    def postprocess(self, values):
        return values


class NMSEErrorMetricBase(ErrorMetric):
    def __init__(self, T, dataset=None, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset
        self.T = T

    def default_name(self):
        return 'Normalized mean squared error'

    def calculate(self, estimator_index, regressor, N_index, N, i):
        dataset = self.dataset

        if regressor.initial_N() is not None and N < regressor.initial_N():
            return np.nan

        if regressor.dataset() is not None:
            dataset = regressor.dataset()

        X_train = dataset.get_train_set_x(N_index, i)
        Y_train = dataset.get_train_set_y(N_index, i)
        X_test = dataset.get_test_set_x(N_index, i)
        Y_test = dataset.get_test_set_y(N_index, i)

        E = dataset.get_edges(N_index, i)
        E_yx = dataset.get_edges_yx(N_index, i)
        E_yy = dataset.get_edges_yy(N_index, i)

        reg_coef = self._regress(estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy)

        Y_hat = reg_coef @ X_test

        error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2) / np.mean(np.linalg.norm(Y_test, axis=0) ** 2)
        return error

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        raise NotImplementedError('This method must be overridden in a derived class')


class JointRegressionNMSEErrorMetric(NMSEErrorMetricBase):
    def __init__(self, T, **kwargs):
        super().__init__(T=T, **kwargs)

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        reg_coef = regressor.regress_joint(
            np.vstack((X_train, Y_train)), E, self.T)

        return reg_coef


class ConditionalRegressionNMSEErrorMetric(NMSEErrorMetricBase):
    def __init__(self, T, **kwargs):
        super().__init__(T=T, **kwargs)

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        reg_coef = regressor.regress_conditional(
            X_train, Y_train, E_yx, E_yy, self.T)

        return reg_coef
