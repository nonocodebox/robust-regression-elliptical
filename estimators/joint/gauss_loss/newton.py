import numpy as np
from ..joint import JointEstimator
import util


class GMRFoptimizer(object):
    """
    TODO: Cite source (github)
    """
    def __init__(self, d, edge_indices):
        self.K = np.eye(d)
        self.K_inv = np.eye(d)
        self.d = d
        self.S = None
        self.edges = edge_indices
        self.learning_rate = 10.
        # step size reduction factor for line search
        self.beta = 0.5
        self.slack = 0.05

    def set_K(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def line_search(self, direction):
        # returns cholesky decomposition of Lambda and the learning rate
        alpha = self.learning_rate
        while True:
            new_point = self.K + alpha * direction
            if not np.isfinite(new_point).all():
                alpha = alpha * self.beta
                continue
            pd, L = util.check_pd(new_point)
            if pd and self.check_descent(direction, alpha):
                # step is positive definite and we have sufficient descent
                break
                # TODO maybe want to return newt+alpha, to reuse computation
            alpha = alpha * self.beta
        return L, alpha

    def check_descent(self, direction, alpha):
        # check if we have made suffcient descent
        grad_K = self.S - self.K_inv

        DK = np.trace(np.dot(grad_K, direction))

        nll_a = self.neg_log_likelihood_wrt_K(self.K + alpha * direction)
        nll_b = self.neg_log_likelihood_wrt_K(self.K) + alpha * self.slack * DK
        return nll_a <= nll_b

    def neg_log_likelihood(self):
        if self.S is None:
            return None
        else:
            return np.trace(self.S.dot(self.K)) - util.log(np.linalg.det(self.K))

    def neg_log_likelihood_wrt_K(self, cand_K):
        # compute the negative log-likelihood of the GCRF when Theta is fixed
        return -util.log(np.linalg.det(cand_K)) + np.trace(np.dot(self.S, cand_K))

    def descent_direction_K(self, max_iter=1):
        delta = np.zeros_like(self.K)
        U = np.zeros_like(self.K)

        Sigma = self.K_inv

        for _ in range(max_iter):
            for i, j in np.random.permutation(np.array(self.edges)):
                if i > j:
                    # seems ok since we look for upper triangular indices in active set
                    continue

                if i == j:
                    a = Sigma[i, i] ** 2
                else:
                    a = Sigma[i, j] ** 2 + Sigma[i, i] * Sigma[j, j]

                b = self.S[i, j] - Sigma[i, j] + np.dot(Sigma[i, :], U[:, j])

                if i == j:
                    u = -b / a
                    delta[i, i] += u
                    U[i, :] += u * Sigma[i, :]
                else:
                    u = -b / a
                    delta[j, i] += u
                    delta[i, j] += u
                    U[j, :] += u * Sigma[i, :]
                    U[i, :] += u * Sigma[j, :]

        return delta

    def reset_K_estimates(self):
        self.K = np.eye(self.d)
        self.K_inv = np.eye(self.d)

    def alt_newton_coord_descent(self, X, max_iter=200, convergence_tolerance=1e-6):
        m = X.shape[1]
        self.S = X.dot(X.T) / m
        self.nll = []
        self.lrs = []

        converged_up_to_tolerance = False
        for t in range(max_iter):
            if t % 100 == 0:
                print('newton_iter {}='.format(X.shape[1]), t)
            # update variable params
            self.nll.append(self.neg_log_likelihood())

            # solve D_lambda via coordinate descent
            K_direction = self.descent_direction_K()
            if not np.isfinite(K_direction).all():
                EPS = 1e-05
                self.K_inv = np.linalg.inv(self.K + EPS * np.eye(self.d))
                K_direction = self.descent_direction_K()
                if not np.isfinite(K_direction).all():
                    print('Newton optimization failed due to overflow.')
                    return self.K.copy(), converged_up_to_tolerance

            # line search for best step size
            learning_rate = self.learning_rate
            LL, learning_rate = self.line_search(K_direction)
            self.lrs.append(learning_rate)

            prev_K = np.array(self.K)
            self.K = self.K.copy() + learning_rate * K_direction

            # update variable params
            # use chol decomp from the backtracking
            self.K_inv = util.chol_inv(LL)
            if not np.isfinite(self.K_inv).all():
                EPS = 1e-05
                self.K_inv = np.linalg.inv(self.K + EPS * np.eye(self.d))
                if not np.isfinite(self.K_inv).all():
                    print('Newton optimization failed due to overflow.')
                    return self.K.copy(), converged_up_to_tolerance

            if t > 0 and np.abs(self.nll[-1] - self.neg_log_likelihood()) < convergence_tolerance:
                converged_up_to_tolerance = True
                break
        return self.K.copy(), converged_up_to_tolerance


class NewtonJointEstimator(JointEstimator):
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
        return 'Joint Newton'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        p = X.shape[0]

        opt = GMRFoptimizer(p, E)

        if K_0 is not None:
            opt.set_K(K_0)
        K, _ = opt.alt_newton_coord_descent(X, max_iter=self.newton_num_steps, convergence_tolerance=self.newton_tol)

        return K
