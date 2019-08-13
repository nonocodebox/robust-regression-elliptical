import numpy as np
from ..joint import JointEstimator
from ..gauss_loss.newton import GMRFoptimizer
import util


class MMNewtonJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using Minimization-Majorization with Newton's Algorithm.
    """

    def __init__(self, loss, max_iters=1000, tolerance=1e-6,
                 newton_num_steps=750, newton_tol=1e-6, **kwargs):
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
        return 'MM Newton Joint'

    def _scale_dataset(self, X, K):
        """
        Scale each example in the dataset by sqrt(g_grad(x^\top K x))
        :param X: The input data.
        :param K: The estimated inverse covariance matrix.
        :return The scaled input data.
        """
        [d, m] = X.shape
        z_vec = np.diag(X.T.dot(K).dot(X))
        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        return X_scaled

    def _elliptical_objective(self, X, K):
        """
        Elliptical objective function.
        :param X: The input data.
        :param K: The estimated inverse covariance matrix.
        :return: The objective value.
        """
        [d, m] = X.shape
        z_vec = np.diag(X.T.dot(K).dot(X))
        mean_g_of_z = np.mean([self.loss.func(z) for z in z_vec])
        return mean_g_of_z - util.log(np.linalg.det(K))

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        [d, m] = X.shape
        K = K_0 if K_0 is not None else np.eye(d)

        scaled_X = self._scale_dataset(X, K)
        gmrf_optimizer = GMRFoptimizer(d, E)
        gmrf_optimizer.set_K(K)
        converged_up_to_tolerance = False

        K, _ = gmrf_optimizer.alt_newton_coord_descent(scaled_X, max_iter=self.newton_num_steps,
                                                       convergence_tolerance=self.newton_tol)
        prev_objective = self._elliptical_objective(X, K)
        for t in np.arange(self.max_iters - 1):
            scaled_X = self._scale_dataset(X, K)
            #     print('effect of scaling: {}'.format(np.linalg.norm(scaled_X-X)))
            K, _ = gmrf_optimizer.alt_newton_coord_descent(scaled_X, max_iter=self.newton_num_steps,
                                                           convergence_tolerance=self.newton_tol)
            cur_objective = self._elliptical_objective(X, K)
            if (np.abs(cur_objective - prev_objective) < self.tolerance):
                converged_up_to_tolerance = True
                return K
            else:
                prev_objective = cur_objective
        return K


class MMJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using Minimization-Majorization.
    """

    def __init__(self, joint_estimator, loss, max_iters=1000, tolerance=1e-6, **kwargs):
        """
        Initialize the estimator.
        :param joint_estimator: The joint estimator to use in each MM step.
        :param loss: The loss to use. Use losses.* package.
        :param max_iters: Maximum number of iterations.
        :param tolerance: The convergence tolerance.
        """
        super().__init__(**kwargs)

        self.estimator = joint_estimator
        self.loss = loss

        self.max_iters = max_iters
        self.tolerance = tolerance

    def default_name(self):
        return 'MM ' + self.estimator.name()

    def _scale_dataset(self, X, K):
        """
        Scale each example in the dataset by sqrt(g_grad(x^\top K x))
        :param X: The input data.
        :param K: The estimated inverse covariance matrix.
        :return The scaled input data.
        """
        [d, m] = X.shape
        z_vec = np.diag(X.T.dot(K).dot(X))
        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        return X_scaled

    def _elliptical_objective(self, X, K):
        """
        Elliptical objective function.
        :param X: The input data.
        :param K: The estimated inverse covariance matrix.
        :return: The objective value.
        """
        [d, m] = X.shape
        z_vec = np.diag(X.T.dot(K).dot(X))
        mean_g_of_z = np.mean([self.loss.func(z) for z in z_vec])
        return mean_g_of_z - util.log(np.linalg.det(K))

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        [d, m] = X.shape
        K = K_0 if K_0 is not None else np.eye(d)

        scaled_X = self._scale_dataset(X, K)

        converged_up_to_tolerance = False

        K = self.estimator.estimate_joint(scaled_X, E, T, K_0=K)

        prev_objective = self._elliptical_objective(X, K)
        for t in np.arange(self.max_iters - 1):
            scaled_X = self._scale_dataset(X, K)
            K = self.estimator.estimate_joint(scaled_X, E, T, K_0=K)
            cur_objective = self._elliptical_objective(X, K)
            if np.abs(cur_objective - prev_objective) < self.tolerance:
                converged_up_to_tolerance = True
                return K
            else:
                prev_objective = cur_objective
        return K
