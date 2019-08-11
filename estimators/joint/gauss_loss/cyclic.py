from ..joint import JointEstimator
import numpy as np
import util


class SecondCyclicJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using the second cyclic algorithm.
    Notations are according to the paper.
    TODO: Cite paper
    """

    def __init__(self, **kwargs):
        """
        Initializes the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Second cyclic'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure.
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        L = util.sample_covariance(X)
        p = X.shape[0]

        # Lower triangle, not including the main diagonal.
        C = [(i, j) for i in range(p) for j in range(i) if (i, j) in E]
        m = len(C)

        # Cache for mm call results
        mm_cache = {}

        def mm(M, a, b=None):
            """
            e.g M_a,b where a = {0,1,2}, p = 4 then M_a,b =M[{0,1,2}, {3}].
            M_a = M[{0,1,2}, {0,1,2}].
            """
            if b is None:
                b = a

            # Use cache since meshgrid is a heavy operation. Check if it is already
            # in cache. If not call meshgrid.
            cached = mm_cache.get((a, b), None)

            if not cached:
                x, y = np.meshgrid(b.I, a.I)
                mm_cache[(a, b)] = x, y
            else:
                x, y = cached

            return M[y, x]

        def inv2(M):
            """
            Closed form of inverse of 2x2 matrix.
            Faster than calling np.linalg.pinv()
            """
            (a, b), (c, d) = M
            return (1 / (a * d - b * c)) * np.array([[d, -b], [-c, a]])

        def inv(M):
            if M.shape == (2, 2):
                return inv2(M)

            return np.linalg.pinv(M)

        def Q_19(R, a, B):
            if not isinstance(a, SecondCyclicJointEstimator._IndexSet):
                a = SecondCyclicJointEstimator._IndexSet(a, p)

            # Build matirx blocks that build the output.
            B_a = mm(B, a)
            R_ai = inv(mm(R, a))
            R_ata = mm(R, a.C, a)
            R_aat = mm(R, a, a.C)
            R_at = mm(R, a.C)

            # np.vstack+np.hstack run faster than np.block.
            # hstack builds matrix blocks horizontally.
            Q = np.vstack((
                np.hstack((
                    B_a,
                    util.matmul(B_a, R_ai, R_aat)
                )),
                np.hstack((
                    util.matmul(R_ata, R_ai, B_a),
                    R_at - util.matmul(R_ata, R_ai, np.eye(a.n) - util.matmul(B_a, R_ai), R_aat)
                ))
            ))

            # Scatter matrix according to a's indices.
            return util.reorder_array(Q, a.I + a.C.I)

        def Q_18(R, a, B):
            R_ai = np.linalg.pinv(mm(R, a))
            B_ai = np.linalg.pinv(mm(B, a))
            R_i = np.linalg.pinv(R)

            Q_i = R_i + util.reorder_array(np.pad(B_ai - R_ai, ((0, a.C.n),), 'constant'), a.I + a.C.I)

            return np.linalg.pinv(Q_i)

        def Y(e, K):
            return Q_19(R=K, a=e, B=L)

        # As denoted in the paper M is the initial guess.
        M = K_0 if K_0 is not None else np.eye(p)

        EPS = 1e-8
        prevK = None
        K = np.linalg.pinv(M)

        # Each edge ie index set. This is why we built C at the first place since
        # {1,2} = {2,1}
        edge_sets = [SecondCyclicJointEstimator._IndexSet(e, p) for e in C]

        for s in range(T):
            for e in edge_sets:
                K = Y(e, K)

        return np.linalg.pinv(K)  # Return Q^-1

    class _IndexSet(object):
        """
        Represents set of indices of a matrix, according to the notation in the
        paper.
        """

        def __init__(self, indices, p):
            if not isinstance(indices, (list, tuple)):
                indices = (indices,)

            self.indices = tuple(set(indices))
            self.p = p
            self.n = len(self.indices)
            # Initialize the complementary set to None.
            self._C = None

        @property
        def I(self):
            return self.indices

        @property
        def C(self):
            """
            Complementary set e.g M_a,a_c = M[a, a_c]
            """
            if self._C is None:
                indices = tuple(i for i in range(self.p) if i not in self.indices)
                self._C = SecondCyclicJointEstimator._IndexSet(indices, self.p)

            return self._C

        def __str__(self):
            # Useful for debugging pupuses.
            return str(self.I) + ', p=' + str(self.p)


class InvestJointEstimator(JointEstimator):
    """
    Inverse covariance estimator using the invest algorithm (first cyclic).
    Translated from Fortran.
    TODO: Cite paper
    """
    def __init__(self, **kwargs):
        """
        Initializes the estimator.
        """
        super().__init__(**kwargs)

    def default_name(self):
        return 'Invest'

    def estimate_joint(self, X, E, T, K_0=None):
        """
        Returns estimated inverse covariance.
        :param X: Data matrix of size (number of features, number of samples)
        :param E: Prior structure.
        :param T: Maximum number of iterations.
        :param K_0: Initial value for the estimated matrix.
        :return: Estimated inverse covariance matrix.
        """
        p = X.shape[0]
        EPS = 1e-8

        t = 0
        prevK_t = None
        converged = False

        # invest expects j > i.
        not_E_upper_triangle = [(i, j) for i in range(p) for j in range(p) if (i,j) not in E and j > i]

        K_t = np.linalg.pinv(util.sample_covariance(X)) if K_0 is None else K_0

        while not converged:
            for i, j in not_E_upper_triangle:
                self._invest_edge(K_t, i, j)

            converged = (prevK_t is not None and
                util.l1_constraint_value(K_t, E) <= EPS) or t >= T

            prevK_t = np.copy(K_t)
            t += 1

        K_t = np.triu(K_t) + np.tril(K_t.T, -1)

        return K_t

    @staticmethod
    def _invest_edge(mat, i, j):
        """
        Implementation of the invest algorithm for a single edge, as described in the paper.
        This method updates the matrix in-place.
        Translated from Fortran.
        :param mat: The input matrix
        :param i: The edge's row (i < j).
        :param j: The edge's column (i < j).
        """
        ndim, nvar = mat.shape

        i += 1
        j += 1

        # Check parameters
        if i < 1 or j <= i or j > nvar:
            print('i={}, j={}, nvar={}'.format(i, j, nvar))
            raise ValueError('Incorrect parameters')

        # Set counters
        i1 = i - 1
        i2 = i + 1
        j1 = j - 1
        j2 = j + 1

        # Store values
        mii = mat[i - 1, i - 1]
        mjj = mat[j - 1, j - 1]
        mij = mat[i - 1, j - 1]
        d = mii * mjj - mij ** 2

        # Positions (i, i), (j, j), (i, j)
        mat[i - 1, i - 1] = d / mjj
        mat[j - 1, j - 1] = d / mii
        mat[i - 1, j - 1] = 0.

        # Reset values
        mii = mij / mii
        mjj = mij / mjj
        mij = -mij / d

        # Positions with K and L less than i
        if i != 1:
            for k in range(1, i1+1):
                save1 = mat[k-1, i-1]
                save2 = mat[k-1, j-1]
                mat[k-1, i-1] = save1 - mjj * save2
                mat[k-1, j-1] = save2 - mii * save1
                mat[k-1, k-1] += mij * (mat[k-1, i-1] * save2 + mat[k-1, j-1] * save1)

                if k == i1:
                    break

                k1 = k + 1
                for l in range(k1, i1+1):
                    mat[k-1, l-1] += mij * (mat[k-1, i-1] * mat[l-1, j-1] + mat[k-1, j-1] * mat[l-1, i-1])

            # 1
            # Positions with K less than i and L between i and j
            # 2
            if i2 != j:
                for k in range(1, i1+1):
                    for l in range(i2, j1+1):
                        mat[k-1, l-1] += mij * (mat[k-1, i-1] * mat[l-1, j-1] + mat[k-1, j-1] * mat[i-1, l-1])
                # 3
            # Positions with K less than i and L greater than j
            # 4
            if j != nvar:
                for k in range(1, i1+1):
                    for l in range(j2, nvar+1):
                        mat[k-1, l-1] += mij * (mat[k-1, i-1] * mat[j-1, l-1] + mat[k-1, j-1] * mat[i-1, l-1])
                # 5
        # Positions with K and L between i and j
        # 6
        if i2 != j:
            for k in range(i2, j1+1):
                save1 = mat[i-1, k-1]
                save2 = mat[k-1, j-1]
                mat[i-1, k-1] = save1 - mjj * save2
                mat[k-1, j-1] = save2 - mii * save1
                mat[k-1, k-1] += mij * (mat[i-1, k-1] * save2 + mat[k-1, j-1] * save1)

                if k == j1:
                    break

                k1 = k + 1
                for l in range(k1, j1+1):
                    mat[k-1, l-1] += mij * (mat[i-1, k-1] * mat[l-1, j-1] + mat[k-1, j-1] * mat[i-1, l-1])
            # 7
            # Positions with K between i and j and L greater than j
            # 8
            if j != nvar:
                for k in range(i2, j1+1):
                    for l in range(j2, nvar+1):
                        mat[k-1, l-1] += mij * (mat[i-1, k-1] * mat[j-1, l-1] + mat[k-1, j-1] * mat[i-1, l-1])
                # 9
        # Positions with K and L greater than j
        # 10
        if j != nvar:
            for k in range(j2, nvar+1):
                save1 = mat[i-1, k-1]
                save2 = mat[j-1, k-1]
                mat[i-1, k-1] = save1 - mjj * save2
                mat[j-1, k-1] = save2 - mii * save1
                mat[k-1, k-1] += mij * (mat[i-1, k-1] * save2 + mat[j-1, k-1] * save1)

                if k == nvar:
                    break

                k1 = k + 1
                for l in range(k1, nvar+1):
                    mat[k-1, l-1] += mij * (mat[i-1, k-1] * mat[j-1, l-1] + mat[j-1, k-1] * mat[i-1, l-1])
            # 11
        # 12
