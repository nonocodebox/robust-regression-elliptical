import numpy as np
from ..conditional import ConditionalEstimator
import util


class GCRFoptimizer():
    def __init__(self, nx, ny, edge_indices_Kyy, edge_indices_Kyx):
        self.Kyy = np.eye(ny)
        self.Kyy_inv = np.eye(ny)
        self.Kyx = np.zeros((ny, nx))
        self.nx = nx
        self.ny = ny
        self.Sxx = None
        self.Sxy = None
        self.Syy = None
        self.yy_edges = edge_indices_Kyy
        self.yx_edges = edge_indices_Kyx
        self.learning_rate = 0.5
        # step size reduction factor for line search
        self.beta = 0.5
        self.slack = 0.05

    def set_Kyy(self, Kyy):
        self.Kyy = Kyy
        self.Kyy_inv = np.linalg.inv(Kyy)

    def set_Kyx(self, Kyx):
        self.Kyx = Kyx

    def line_search(self, direction):
        # returns cholesky decomposition of Lambda and the learning rate
        alpha = self.learning_rate
        while True:
            EPS = 1e-5
            pd, L = util.check_pd(self.Kyy + alpha * direction - EPS * np.eye(self.ny))
            if pd and self.check_descent(direction, alpha):
                # step is positive definite and we have sufficient descent
                break
                # TODO maybe want to return newt+alpha, to reuse computation
            alpha = alpha * self.beta
            # if alpha < 0.1:
            #   return L, alpha
        return L, alpha

    def check_descent(self, direction, alpha):
        # check if we have made suffcient descent
        grad_Kyy = self.Syy - self.Kyy_inv - \
                   self.Kyy_inv.dot(self.Kyx).dot(self.Sxx).dot(self.Kyx.T).dot(self.Kyy_inv)

        DKyy = np.trace(np.dot(grad_Kyy, direction))

        nll_a = self.neg_log_likelihood_wrt_Kyy(self.Kyy + alpha * direction)
        nll_b = self.neg_log_likelihood_wrt_Kyy(self.Kyy) + alpha * self.slack * DKyy
        return nll_a <= nll_b

    def neg_log_likelihood(self):
        if self.Sxx is None or self.Syy is None or self.Sxy is None:
            return None
        else:
            return np.trace(self.Syy.dot(self.Kyy) + \
                            2 * self.Sxy.T.dot(self.Kyx.T) + \
                            self.Kyx.dot(self.Sxx.dot(self.Kyx.T)).dot(self.Kyy_inv)) \
                   - util.log(np.linalg.det(self.Kyy))

    def neg_log_likelihood_wrt_Kyy(self, cand_Kyy):
        # compute the negative log-likelihood of the GCRF when Theta is fixed
        cand_Kyy_inv = np.linalg.inv(cand_Kyy)
        X_quad_term = cand_Kyy_inv.dot(self.Kyx.dot(self.Sxx).dot(self.Kyx.T)).dot(cand_Kyy_inv)
        return -util.log(np.linalg.det(cand_Kyy)) + \
               np.trace(np.dot(self.Syy, cand_Kyy) + \
                        np.dot(X_quad_term, cand_Kyy))

    def descent_direction_Kyy(self, max_iter=1):
        delta = np.zeros_like(self.Kyy)
        U = np.zeros_like(self.Kyy)

        Sigma_yy = self.Kyy_inv
        X_quad_term = Sigma_yy.dot(self.Kyx.dot(self.Sxx).dot(self.Kyx.T)).dot(Sigma_yy)

        for _ in range(max_iter):
            for i, j in np.random.permutation(np.array(self.yy_edges)):
                if i > j:
                    # seems ok since we look for upper triangular indices in active set
                    continue

                if i == j:
                    a = Sigma_yy[i, i] ** 2 + 2 * Sigma_yy[i, i] * X_quad_term[i, i]
                else:
                    a = (Sigma_yy[i, j] ** 2 + Sigma_yy[i, i] * Sigma_yy[j, j] +
                         Sigma_yy[i, i] * X_quad_term[j, j] + 2 * Sigma_yy[i, j] * X_quad_term[i, j] +
                         Sigma_yy[j, j] * X_quad_term[i, i])

                b = (self.Syy[i, j] - Sigma_yy[i, j] - X_quad_term[i, j] +
                     np.dot(Sigma_yy[i, :], U[:, j]) +
                     np.dot(X_quad_term[i, :], U[:, j]) +
                     np.dot(X_quad_term[j, :], U[:, i]))

                if i == j:
                    u = -b / a
                    delta[i, i] += u
                    U[i, :] += u * Sigma_yy[i, :]
                else:
                    u = -b / a
                    delta[j, i] += u
                    delta[i, j] += u
                    U[j, :] += u * Sigma_yy[i, :]
                    U[i, :] += u * Sigma_yy[j, :]

        return delta

    def Kyx_coordinate_descent(self, max_iter=1):
        V = np.dot(self.Kyx.T, self.Kyy_inv)
        for _ in range(max_iter):
            for i, j in np.array(self.yx_edges):
                a = 2 * self.Kyy_inv[i, i] * self.Sxx[j, j]
                b = 2 * self.Sxy[j, i] + 2 * np.dot(self.Sxx[j, :], V[:, i])

                u = -b / a

                self.Kyx[i, j] += u
                V[j, :] += u * self.Kyy_inv[i, :]

        return self.Kyx

    def reset_K_estimates(self):
        self.Kyy = np.eye(self.ny)
        self.Kyy_inv = np.eye(self.ny)
        self.Kyx = np.zeros((self.ny, self.nx))

    def alt_newton_coord_descent(self, X, Y, max_iter=200, convergence_tolerance=1e-6):
        m = X.shape[1]
        self.Sxx = X.dot(X.T) / m
        self.Syy = Y.dot(Y.T) / m
        self.Sxy = X.dot(Y.T) / m

        self.nll = []
        self.lnll = []
        self.lrs = []

        converged_up_to_tolerance = False
        for t in range(max_iter):
            if t % 100 == 0:
                print('newton_iter {}='.format(X.shape[1]), t)
            # update variable params
            self.nll.append(self.neg_log_likelihood())

            # solve D_lambda via coordinate descent
            Kyy_direction = self.descent_direction_Kyy()
            if not np.isfinite(Kyy_direction).all():
                print('Newton optimization failed due to overflow.')
                return self.Kyy.copy(), self.Kyx.copy(), converged_up_to_tolerance

            # line search for best step size
            learning_rate = self.learning_rate
            LL, learning_rate = self.line_search(Kyy_direction)
            self.lrs.append(learning_rate)

            prev_Kyy = np.array(self.Kyy)
            self.Kyy = self.Kyy.copy() + learning_rate * Kyy_direction

            # update variable params
            self.Kyy_inv = util.chol_inv(LL)  # use chol decomp from the backtracking

            # solve theta
            prev_Kyx = np.array(self.Kyx)
            self.Kyx = self.Kyx_coordinate_descent()

            if not (np.isfinite(self.Kyy_inv).all() and np.isfinite(self.Kyx).all()):
                EPS = 1e-05
                self.Kyy_inv = np.linalg.inv(self.Kyy + EPS * np.eye(self.ny))
                if not np.isfinite(self.Kyy_inv).all():
                    print('Newton optimization failed due to overflow.')
                    return self.Kyy.copy(), self.Kyx.copy(), converged_up_to_tolerance

            if t > 0 and np.abs(self.nll[-1] - self.neg_log_likelihood()) < convergence_tolerance:
                converged_up_to_tolerance = True
                break
        return self.Kyy.copy(), self.Kyx.copy(), converged_up_to_tolerance


class NewtonConditionalEstimator(ConditionalEstimator):
    def __init__(self, newton_tol=1e-6, newton_num_steps=750, **kwargs):
        """
        Initialize the estimator.
        :param newton_tol: Tolerance for newton convergence.
        :param newton_num_steps: Maximum number of steps.
        """
        super().__init__(**kwargs)
        self.newton_tol = newton_tol
        self.newton_num_steps = newton_num_steps

    def default_name(self):
        return 'CRF Newton'

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
        dx = X.shape[0]
        dy = Y.shape[0]

        opt = GCRFoptimizer(dx, dy, E_yy, E_yx)
        if K_yx_0 is not None:
            opt.set_Kyx(K_yx_0)
        if K_yy_0 is not None:
            opt.set_Kyy(K_yy_0)
        K_yy, K_yx, _ = opt.alt_newton_coord_descent(X, Y, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

        return K_yy, K_yx
