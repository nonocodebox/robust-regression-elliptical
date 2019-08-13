from ..joint import JointEstimator
import numpy as np
import util


class ModRegJointEstimator(JointEstimator):
    """
    Implementation of the modified regression algorithm.
    Hastie, T., Tibshirani, R., Friedman, J. (2001). The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc.
    Algorithm 17.1
    """

    def __init__(self, **kwargs):
        """
        Initialize the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Modified Regression'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples).
        :param E: Prior structure. List of tuples, where each tuple represents an edge (row, column).
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        EPS = 1e-12
        p = X.shape[0]

        S = util.sample_covariance(X) if K_0 is None else np.linalg.pinv(K_0)
        Q_t = S.copy()
        betas = np.zeros((p - 1, p))
        prevQ_t = None
        t = 0
        converged = False
        while not converged:
            for j in range(p):
                # Partition the matrix W, without j row and column.
                W_11 = np.delete(Q_t,  (j), axis=0)
                W_11 = np.delete(W_11, (j), axis=1)

                # Remove zero elements in j col. These rows carry no information
                # and can be removed.
                # s_12 = np.delete(S[j,:], (j))
                beta = self._beta_remove_zero_edges(Q_t, S[:, j], E, j) # p-1x 1
                w_12 = np.matmul(W_11, beta)
                betas[:, j] = beta

                # Update Q's row and column j.
                Q_t[:j,j] = w_12[:j]
                Q_t[j+1:,j] = w_12[j:]
                Q_t[j, :j] = w_12[:j]
                Q_t[j,j+1:] = w_12[j:]

            converged = (prevQ_t is not None and
                np.linalg.norm(prevQ_t - Q_t, 'fro') <= EPS) or t >= T

            prevQ_t = np.copy(Q_t)
            t += 1

        K_t = np.zeros((p, p))

        for j in range(p):
            w_12 = np.delete(Q_t[:, j], j)
            k_22 = 1.0 / (S[j, j] - np.dot(w_12, betas[:, j]))
            k_12 = -betas[:, j] * k_22
            K_t[j, j] = k_22
            K_t[:j, j] = k_12[:j]
            K_t[j+1:, j] = k_12[j:]

        # print('T=', T, ' t=', t-1)

        return K_t

    @staticmethod
    def _beta_remove_zero_edges(W, s, E, j):
        p = W.shape[0]
        # Note this means that for every j should have a neigbor besides itself
        edge_rows = set([a for a,b in E if b == j])
        rows_to_delete = set(range(p)).difference(edge_rows)
        rows_to_delete = rows_to_delete.union({j})
        rows_to_delete = sorted(rows_to_delete, reverse=True)

        for i in rows_to_delete:
            W = np.delete(W, (i), axis=0)
            W = np.delete(W, (i), axis=1)
            s = np.delete(s, (i))

        beta = np.matmul(np.linalg.pinv(W), s) # q-1 x 1

        padded_beta = np.zeros(p)
        b = 0
        for a in range(p):
            if a in rows_to_delete:
                continue
            padded_beta[a] = beta[b]
            b+=1
        return np.delete(padded_beta, (j)) # p-1 x 1
