import numpy as np
from ..conditional import ConditionalEstimator
from ..gauss_loss.newton import GCRFoptimizer


class MMNewtonConditionalEstimator(ConditionalEstimator):
    """
    Inverse covariance estimator using Minimization-Majorization with Newton's Algorithm.
    """

    def __init__(self, loss, max_iters=1000, tolerance=1e-6, newton_num_steps=750, newton_tol=1e-6, **kwargs):
        """
        Initialize the estimator.
        :param loss: The loss to use. Use losses.* package.
        :param max_iters: Maximum number of iterations.
        :param tolerance: The convergence tolerance.
        :param newton_num_steps: The maximum number of steps for each newton execution.
        :param newton_tol: The convergence tolerance for each newton execution.
        """
        super().__init__(**kwargs)

        self.loss = loss

        self.max_iters = max_iters
        self.tolerance = tolerance
        self.newton_num_steps = newton_num_steps
        self.newton_tol = newton_tol

    def default_name(self):
        return 'MM Newton'

    def _scale_conditional_dataset(self, X, Y, K_yy, K_yx):
        """
        Scale each example in the dataset by sqrt(g_grad((y-K_yy^-1*K_yx*x)^\top*K*(y-K_yy^-1*K_yx*x)))
        :param X: The input features.
        :param Y: The targets.
        :param K_yy: The targets estimated inverse covariance matrix.
        :param K_yx: The targets-features estimated inverse covariance matrix.
        :return The scaled input features and targets.
        """
        [d, m] = X.shape
        K_yy_inv = np.linalg.inv(K_yy)
        mean_offsets = Y + K_yy_inv.dot(K_yx).dot(X)  # [ny,m]
        z_vec = np.diag(mean_offsets.T.dot(K_yy).dot(mean_offsets))

        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        Y_scaled = np.multiply(Y, scaling_factors)
        return X_scaled, Y_scaled

    def estimate_conditional(self, X, Y, E_yx, E_yy, T, K_yx_0=None, K_yy_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Input features matrix of size (number of features, number of samples).
        :param Y: Targets matrix of size (number of targets, number of samples).
        :param E_yx: Prior targets-features structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param E_yy: Prior targets structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_yx_0: Initial value for the estimated K_yx matrix.
        :param K_yy_0: Initial value for the estimated K_yy matrix.
        :return: Inverse covariance matrices (K_yx, K_yy)
                 K_yx is the targets-features inverse covariance matrix.
                 K_yy is the targets inverse covariance matrix.
        """
        [nx, m] = X.shape
        [ny, m] = Y.shape

        K_yy = K_yy_0 if K_yy_0 is not None else np.eye(ny)
        K_yx = K_yx_0 if K_yx_0 is not None else np.zeros((ny, nx))

        scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
        edge_indices_yy_with_diag = np.concatenate([E_yy,
                                                    [[i, i] for i in range(ny)]])
        gcrf_optimizer = GCRFoptimizer(nx, ny, edge_indices_yy_with_diag, E_yx)
        gcrf_optimizer.set_Kyy(K_yy)
        gcrf_optimizer.set_Kyx(K_yx)

        converged_up_to_tolerance = False
        K_yy, K_yx, _ = gcrf_optimizer.alt_newton_coord_descent(
            scaled_X, scaled_Y, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

        for t in np.arange(self.max_iters - 1):
            scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
            next_K_yy, next_K_yx, _ = gcrf_optimizer.alt_newton_coord_descent(
                scaled_X, scaled_Y, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

            if (np.linalg.norm(K_yy - next_K_yy) < self.tolerance and np.linalg.norm(K_yx - next_K_yx) < self.tolerance):
                converged_up_to_tolerance = True
                return next_K_yy, next_K_yx
            else:
                K_yy = next_K_yy
                K_yx = next_K_yx

        return K_yy, K_yx


class MMConditionalEstimator(ConditionalEstimator):
    """
    Inverse covariance estimator using Minimization-Majorization.
    """

    def __init__(self, conditional_estimator, loss, max_iters=1000, tolerance=1e-6, **kwargs):
        """
        Initialize the estimator.
        :param conditional_estimator: The conditional estimator to use in each MM step.
        :param loss: The loss to use. Use losses.* package.
        :param max_iters: Maximum number of iterations.
        :param tolerance: The convergence tolerance.
        """
        super().__init__(**kwargs)

        self.estimator = conditional_estimator
        self.loss = loss

        self.max_iters = max_iters
        self.tolerance = tolerance

    def default_name(self):
        return 'MM ' + self.estimator.name()

    def _scale_conditional_dataset(self, X, Y, K_yy, K_yx):
        """
        Scale each example in the dataset by sqrt(g_grad((y-K_yy^-1*K_yx*x)^\top*K*(y-K_yy^-1*K_yx*x)))
        :param X: The input features.
        :param Y: The targets.
        :param K_yy: The targets estimated inverse covariance matrix.
        :param K_yx: The targets-features estimated inverse covariance matrix.
        :return The scaled input features and targets.
        """
        [d, m] = X.shape
        K_yy_inv = np.linalg.inv(K_yy)
        mean_offsets = Y + K_yy_inv.dot(K_yx).dot(X)  # [ny,m]
        z_vec = np.diag(mean_offsets.T.dot(K_yy).dot(mean_offsets))

        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        Y_scaled = np.multiply(Y, scaling_factors)
        return X_scaled, Y_scaled

    def estimate_conditional(self, X, Y, E_yx, E_yy, T, K_yx_0=None, K_yy_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Input features matrix of size (number of features, number of samples).
        :param Y: Targets matrix of size (number of targets, number of samples).
        :param E_yx: Prior targets-features structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param E_yy: Prior targets structure.
                     List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_yx_0: Initial value for the estimated K_yx matrix.
        :param K_yy_0: Initial value for the estimated K_yy matrix.
        :return: Inverse covariance matrices (K_yx, K_yy)
                 K_yx is the targets-features inverse covariance matrix.
                 K_yy is the targets inverse covariance matrix.
        """
        [nx, m] = X.shape
        [ny, m] = Y.shape

        K_yy = K_yy_0 if K_yy_0 is not None else np.eye(ny)
        K_yx = K_yx_0 if K_yx_0 is not None else np.zeros((ny, nx))

        scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
        edge_indices_yy_with_diag = np.concatenate([E_yy,
                                                    [[i, i] for i in range(ny)]])

        converged_up_to_tolerance = False
        K_yy, K_yx = self.estimator.estimate_conditional(
            scaled_X, scaled_Y, E_yx, E_yy, T, K_yx_0=K_yx, K_yy_0=K_yy)

        for t in np.arange(self.max_iters - 1):
            scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
            next_K_yy, next_K_yx = self.estimator.estimate_conditional(
                scaled_X, scaled_Y, E_yx, E_yy, T, K_yx_0=K_yx, K_yy_0=K_yy)

            if (np.linalg.norm(K_yy - next_K_yy) < self.tolerance and np.linalg.norm(K_yx - next_K_yx) < self.tolerance):
                converged_up_to_tolerance = True
                return next_K_yy, next_K_yx
            else:
                K_yy = next_K_yy
                K_yx = next_K_yx

        return K_yy, K_yx
