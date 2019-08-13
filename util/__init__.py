import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from . import data


class Nameable(object):
    """
    A nameable object.
    """
    def __init__(self, name=None, context=None, **kwargs):
        """
        Initialize the nameable object.
        :param name: Object name.
        :param context: User context (may be used for callbacks etc).
        """
        super().__init__(**kwargs)
        self._name = name
        self._context = context

    def name(self):
        """
        Returns the object's name.
        :return: The object's name.
        """
        return self._name or self.default_name()

    def context(self):
        """
        Returns the object's context.
        :return: The object's context.
        """
        return self._context

    def default_name(self):
        """
        Returns a default name for the object.
        Should be implemented by derived classes returning a meaningful default name.
        :return: The object's default name.
        """
        raise NotImplementedError('This method must be implemented in a derived class')


class PlotAdditionalParameters(object):
    """
    Class containing additional parameters for assisting plots.
    """
    def __init__(self, initial_N=None, dataset=None, **kwargs):
        """
        Initializes the parameters.
        :param initial_N: The initial N (sample size) value to use. None if unused.
        :param dataset: The plot-specific dataset to use. None if unused.
        """
        super().__init__(**kwargs)
        self._initial_N = initial_N
        self._dataset = dataset

    def initial_N(self):
        """
        Gets the initial N (sample size) value. May be None if unused.
        This is the initial sample size out of different sample sizes used as the X axis.
        :return: The initial N.
        """
        return self._initial_N

    def dataset(self):
        """
        Gets the dataset to use specifically for this plot. May be None if unused.
        :return: The dataset to use.
        """
        return self._dataset


def matmul(*args):
    '''
    Multiplies multiple arrays using matrix multiplication.
    '''
    if len(args) < 1:
        raise Exception('Need at least one array')

    M = args[0]

    for a in args[1:]:
        M = np.matmul(M, a)

    return M


def sample_covariance(X):
    '''
    Returns the argmax of maximum likelihood of gaussian data.
    '''
    n = X.shape[1]
    return np.dot(X, X.T) / n


def slice_array(rows, cols, M):
    '''
    Slices an array given row and column indices.
    :param rows: Array of row indices
    :param cols: Array of column indices
    :return: The sliced array
    '''
    tiled_rows = np.tile(np.array(rows).reshape(-1,1), (1, len(cols)))
    tiled_cols = np.tile(np.array(cols).reshape(1,-1), (len(rows), 1))
    return M[tiled_rows, tiled_cols]

def reorder_array(M, rows, cols=None, inv_indices=True):
    '''
    Reorders an array's rows and columns.
    :param rows: The row order (indices).
    :param cols: The column order (indices). If None, the row order will be used.
    :param inv_indices: Whether to invert the index direction.
                        If False, each index represents an index in the source array.
                        If True, each index represents an index in the destination array.
    :return: The reordered array.
    '''

    if cols is None:
        cols = rows

    # a's matix values are in the first rows and cols. We want to put a's values
    # in the right places. The best way to explain it is by an example. Let's
    # rows = (2,4,...) where a = 2,4. This means that 0->2 and 1->4. Then, what
    # we want is M[r] where r = (_ ,_, 0, _, 1, ... ). This is why we invert
    # the indices.
    if inv_indices:
        rows = [rows.index(i) for i in range(len(rows))]
        cols = [cols.index(i) for i in range(len(cols))]

    M = M[rows, :]
    return M[:, cols]


def l1_constraint_value(K, E):
    p = K.shape[0]
    l1_value = 0
    for i in range(p):
        for j in range(p):
            if (i, j) in E and (j <= i):
                continue
            l1_value += np.abs(K[i,j])
    return l1_value


def diff_from_sc_value(K, S, E):
    Q = np.linalg.pinv(K)
    p = K.shape[0]
    l_1_diff = 0
    for i in range(p):
        for j in range(p):
            if (i, j) in E and (j <= i):
                l_1_diff += np.abs(S[i,j]-Q[i,j])
    return l_1_diff


def normalize_data(X):
    return X / np.sqrt(np.sum(X ** 2, axis=0))  # angular


def split_edges(E, dx, dy):
    E_xx, E_yx, E_yy = [], [], []

    for i, j in E:
        if i < dx and j < dx:
            E_xx.append((i, j))
        elif i >= dx and j >= dx:
            E_yy.append((i - dx, j - dx))
        elif i >= dx and j < dx:
            E_yx.append((i - dx, j))

    return E_xx, E_yx, E_yy


def check_pd(A, lower=True):
    """
    Checks if A is PD.
    If so returns True and Cholesky decomposition,
    otherwise returns False and None
    """
    try:
        return True, np.tril(cho_factor(A, lower=lower)[0])
    except LinAlgError as err:
        if 'not positive definite' in str(err):
            return False, None


def chol_inv(B, lower=True):
    """
    Returns the inverse of matrix A, where A = B*B.T,
    ie B is the Cholesky decomposition of A.
    Solves Ax = I
    given B is the cholesky factorization of A.
    """
    return cho_solve((B, lower), np.eye(B.shape[0]))


def spd_inv(A):
    """
    Inversion of a SPD matrix using Cholesky decomposition.
    """
    return chol_inv(check_pd(A)[1])


def log(x):
    return np.log(x) if x > 0 else -np.inf
