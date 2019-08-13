from ..joint import JointEstimator
import util
import numpy as np


class IPFJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using iterative proportional fitting (IPF).
    TODO: Cite book (ELSII)
    """
    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'IPF'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        EPS = 1e-18
        p = X.shape[0]

        K_t = np.eye(p)
        S = util.sample_covariance(X) if K_0 is None else np.linalg.pinv(K_0)

        # {a,b} = {b, a}
        neighbours_ordered = set([(a,b) for a,b in E if a < b])
        neighbours_ordered = sorted(neighbours_ordered)
        t = 0
        converged = False
        while not converged:
            print('t=', t)
            for i,j in neighbours_ordered:
                setij = [i,j]
                notij = list(set(range(p)) - set(setij))

                # By invertion matrix block theorem.
                K_B = util.slice_array(setij, notij, K_t)
                invK_D = np.linalg.pinv(util.slice_array(notij, notij, K_t))
                K_C = util.slice_array(notij, setij, K_t)
                invS_ij = np.linalg.pinv(util.slice_array(setij, setij, S))
                K_ij = invS_ij + np.matmul(K_B, np.matmul(invK_D, K_C))

                rows = setij
                cols = setij

                # An alternative to meshgrid. Take rows list, make it a coloum and
                # duplicate it col times.
                tiled_rows = np.tile(np.array(rows).reshape(-1,1), (1, len(cols)))
                tiled_cols = np.tile(np.array(cols).reshape(1,-1), (len(rows), 1))
                K_t[tiled_rows, tiled_cols] = K_ij

            #converged = (prevQ_t is not None and
            #    np.linalg.norm(prevQ_t - Q_t, 'fro') <= EPS) or t >= T
            converged = (self._sc_resemblence_error(X, E, K_t) <= EPS) or t >= T

            t += 1

        return K_t

    @staticmethod
    def _sc_resemblence_error(X, E, K):
        """
        Error calculated based on resemblence to sample covariance.
        Used for convergence check (stopping condition).
        :param X: Input data
        :param E: Prior structure
        :param K: Estimated inverse covariance
        :return: SC resemblence error
        """
        S = util.sample_covariance(X)
        error = 0.0
        Q = np.linalg.pinv(K)
        for i,j in E:
            setij = [i,j]
            diff_mat = util.slice_array(setij, setij, S) - util.slice_array(setij, setij, Q)
            error += np.linalg.norm(diff_mat, 'fro')
        return error
