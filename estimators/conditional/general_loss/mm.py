import numpy as np
from ..conditional import ConditionalEstimator
from ..gauss_loss.newton import GCRFoptimizer


class MMNewtonConditionalEstimator(ConditionalEstimator):
    def __init__(self, loss, max_iters=1000, tolerance=1e-6, newton_num_steps=750, newton_tol=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.loss = loss

        self.max_iters = max_iters
        self.tolerance = tolerance
        self.newton_num_steps = newton_num_steps
        self.newton_tol = newton_tol

    def default_name(self):
        return 'MM Newton'

    def _scale_conditional_dataset(self, X, Y, K_yy, K_yx):
        """ Scale each example in the dataset by sqrt(g_grad((y-K_yy^-1*K_yx*x)^\top*K*(y-K_yy^-1*K_yx*x)))"""
        [d, m] = X.shape
        K_yy_inv = np.linalg.inv(K_yy)
        mean_offsets = Y + K_yy_inv.dot(K_yx).dot(X)  # [ny,m]
        z_vec = np.diag(mean_offsets.T.dot(K_yy).dot(mean_offsets))

        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        Y_scaled = np.multiply(Y, scaling_factors)
        return X_scaled, Y_scaled

    def estimate_conditional(self, X, Y, Eyx, Eyy, T, Kyx_0=None, Kyy_0=None):
        [nx, m] = X.shape
        [ny, m] = Y.shape

        K_yy = Kyy_0 if Kyy_0 is not None else np.eye(ny)
        K_yx = Kyx_0 if Kyx_0 is not None else np.zeros((ny, nx))

        scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
        edge_indices_yy_with_diag = np.concatenate([Eyy,
                                                    [[i, i] for i in range(ny)]])
        gcrf_optimizer = GCRFoptimizer(nx, ny, edge_indices_yy_with_diag, Eyx)
        gcrf_optimizer.set_Kyy(K_yy)
        gcrf_optimizer.set_Kyx(K_yx)

        converged_up_to_tolerance = False
        K_yy, K_yx, _ = gcrf_optimizer.alt_newton_coord_descent(
            scaled_X, scaled_Y, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

        for t in np.arange(self.max_iters - 1):
            scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
            next_K_yy, next_K_yx, _ = gcrf_optimizer.alt_newton_coord_descent(
                scaled_X, scaled_Y, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

            #grad_diff = g_grad()
            if (np.linalg.norm(K_yy - next_K_yy) < self.tolerance and np.linalg.norm(K_yx - next_K_yx) < self.tolerance):
                converged_up_to_tolerance = True
                return next_K_yy, next_K_yx
            else:
                K_yy = next_K_yy
                K_yx = next_K_yx

        return K_yy, K_yx


class MMConditionalEstimator(ConditionalEstimator):
    def __init__(self, conditional_estimator, loss, max_iters=1000, tolerance=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.estimator = conditional_estimator
        self.loss = loss

        self.max_iters = max_iters
        self.tolerance = tolerance

    def default_name(self):
        return 'MM ' + self.estimator.name()

    def _scale_conditional_dataset(self, X, Y, K_yy, K_yx):
        """ Scale each example in the dataset by sqrt(g_grad((y-K_yy^-1*K_yx*x)^\top*K*(y-K_yy^-1*K_yx*x)))"""
        [d, m] = X.shape
        K_yy_inv = np.linalg.inv(K_yy)
        mean_offsets = Y + K_yy_inv.dot(K_yx).dot(X)  # [ny,m]
        z_vec = np.diag(mean_offsets.T.dot(K_yy).dot(mean_offsets))

        scaling_factors = np.array([np.sqrt(self.loss.grad(z)) for z in z_vec], ndmin=2)
        X_scaled = np.multiply(X, scaling_factors)
        Y_scaled = np.multiply(Y, scaling_factors)
        return X_scaled, Y_scaled

    def estimate_conditional(self, X, Y, Eyx, Eyy, T, Kyx_0=None, Kyy_0=None):
        [nx, m] = X.shape
        [ny, m] = Y.shape

        K_yy = Kyy_0 if Kyy_0 is not None else np.eye(ny)
        K_yx = Kyx_0 if Kyx_0 is not None else np.zeros((ny, nx))

        scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
        edge_indices_yy_with_diag = np.concatenate([Eyy,
                                                    [[i, i] for i in range(ny)]])

        converged_up_to_tolerance = False
        K_yy, K_yx = self.estimator.estimate_conditional(
            scaled_X, scaled_Y, Eyx, Eyy, T, Kyx_0=K_yx, Kyy_0=K_yy)

        for t in np.arange(self.max_iters - 1):
            scaled_X, scaled_Y = self._scale_conditional_dataset(X, Y, K_yy, K_yx)
            next_K_yy, next_K_yx = self.estimator.estimate_conditional(
                scaled_X, scaled_Y, Eyx, Eyy, T, Kyx_0=K_yx, Kyy_0=K_yy)

            #grad_diff = g_grad()
            if (np.linalg.norm(K_yy - next_K_yy) < self.tolerance and np.linalg.norm(K_yx - next_K_yx) < self.tolerance):
                converged_up_to_tolerance = True
                return next_K_yy, next_K_yx
            else:
                K_yy = next_K_yy
                K_yx = next_K_yx

        return K_yy, K_yx
