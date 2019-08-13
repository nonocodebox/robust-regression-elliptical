from util import Nameable
import numpy as np


class ErrorMetric(Nameable):
    """
    A class for calculating error metrics for plotting.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, estimator_index, estimator, N_index, N, i):
        """
        Calculates the error metric to be plotted.
        :param estimator_index: The estimator's index.
        :param estimator: The estimator object.
        :param N_index: The sample size index in the list of sample sizes.
        :param N: The sample size.
        :param i: The averaging iteration number.
        :return: The calculated error metric to plot.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def postprocess(self, values):
        """
        Post-processes the calculated data after all results are received.
        :param values: The results array, of size (estimators x samples sizes x averaging iterations).
        :return: The postprocessed results array.
        """
        return values


class NMSEErrorMetricBase(ErrorMetric):
    """
    Base class for NMSE metric implementations.
    """

    def __init__(self, T, dataset=None, **kwargs):
        """
        Initialize this metric calculator.
        :param T: The number of averaging iterations.
        :param dataset: The dataset to use. Can be None if the datasets are specified via each
                        estimators's dataset property.
        """
        super().__init__(**kwargs)

        self.dataset = dataset
        self.T = T

    def default_name(self):
        return 'Normalized mean squared error'

    def calculate(self, estimator_index, regressor, N_index, N, i):
        dataset = self.dataset

        # Skip initial N if available
        if regressor.initial_N() is not None and N < regressor.initial_N():
            return np.nan

        # Use the regressor's dataset if available
        if regressor.dataset() is not None:
            dataset = regressor.dataset()

        # Get the data
        X_train = dataset.get_train_set_x(N_index, i)
        Y_train = dataset.get_train_set_y(N_index, i)
        X_test = dataset.get_test_set_x(N_index, i)
        Y_test = dataset.get_test_set_y(N_index, i)

        # Get the edges (structure)
        E = dataset.get_edges(N_index, i)
        E_yx = dataset.get_edges_yx(N_index, i)
        E_yy = dataset.get_edges_yy(N_index, i)

        # Calculate the regression coefficients.
        reg_coef = self._regress(estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy)

        # Predict target values
        Y_hat = reg_coef @ X_test

        # Calculate NMSE
        error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2) / np.mean(np.linalg.norm(Y_test, axis=0) ** 2)
        return error

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        """
        A method for calculating the regression coefficients. Should be implemented in derived classes.
        :param estimator_index: The estimator's index.
        :param estimator: The estimator object.
        :param N_index: The sample size index in the list of sample sizes.
        :param N: The sample size.
        :param i: The averaging iteration number.
        :param X_train: The input features part of the train set.
        :param Y_train: The targets part of the train set.
        :param E: Prior structure.
        :param E_yx: Prior targets-features structure.
        :param E_yy: Prior targets structure.
        :return: The calculated regression coefficients.
        """
        raise NotImplementedError('This method must be overridden in a derived class')


class JointRegressionNMSEErrorMetric(NMSEErrorMetricBase):
    """
    Implementation of the NMSE metric for joint regression.
    """
    def __init__(self, T, **kwargs):
        super().__init__(T=T, **kwargs)

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        reg_coef = regressor.regress_joint(
            np.vstack((X_train, Y_train)), E, self.T)

        return reg_coef


class ConditionalRegressionNMSEErrorMetric(NMSEErrorMetricBase):
    """
    Implementation of the NMSE metric for conditional regression.
    """

    def __init__(self, T, **kwargs):
        super().__init__(T=T, **kwargs)

    def _regress(self, estimator_index, regressor, N_index, N, i, X_train, Y_train, E, E_yx, E_yy):
        reg_coef = regressor.regress_conditional(
            X_train, Y_train, E_yx, E_yy, self.T)

        return reg_coef
