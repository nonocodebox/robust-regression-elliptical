import numpy as np
import plots
import regressors
import estimators
import losses
from datasets import Dataset
import argparse
import statsmodels.sandbox.distributions.multivariate as mdist
from util.data import generate_inverse_covariance_structure, generate_random_sparse_psd, multivariate_generalized_gaussian
from plots.metrics import ErrorMetric
import scipy as sp
import util


AVERAGE_ITERATIONS_DEFAULT = 100
DIMENSION_DEFAULT = 10


def get_elliptic_data(scalar_sampler, n, m_train, K_gt, seed=1):#sparsity_alpha=0.9):
    #np.random.seed(seed)
    num_samples = m_train
    #K_gt = np.float32(get_sparse_high_correlations_K(n, seed,
    #                                                 sparsity_alpha=sparsity_alpha))
    #K_gt = util.data.generate_random_sparse_psd(n, sparsity_alpha)

    inv_K_gt = np.linalg.inv(K_gt)
    if not util.check_pd(inv_K_gt):
        raise Exception('inv_K is NOT PSD')

    spherical_uniform = np.random.randn(n, num_samples)
    spherical_uniform /= np.linalg.norm(spherical_uniform, axis=0)

    scaling_params = scalar_sampler(num_samples)
    X_train = np.multiply(scaling_params.T, sp.linalg.sqrtm(inv_K_gt).dot(spherical_uniform))

    return X_train#, K_gt


class PositiveScalarSamplerFactory():
    """ corresponds to the samples creating a multivariate Gaussian distribution """

    def root_chi_square(self, params):
        sampler = lambda m: np.sqrt(np.random.chisquare(params['df'], size=(m, 1)))
        return sampler

    def exponential(self, params):
        sampler = lambda m: np.random.exponential(params['scale'], size=(m, 1))
        return sampler

    def laplace(self, params):
        sampler = lambda m: np.sqrt(np.random.laplace(scale=params['scale'],
                                                      size=(m, 1)) ** 2)
        return sampler

    def multivariate_t(self, params):
        d = params['dim']
        nu = params['nu']
        sampler = lambda m: np.sqrt(d * np.random.f(d, nu, m))
        return sampler

    def generalized_gaussian(self, params):
        """Returns a sampler for a 2*beta'th root of a Gamma distribution.

        When incorporated as the scalar sampler of an Elliptical distribution, this
        creates the Generalized Gausian distribution, from:
        Pascal et al. - Parameter Estimation For Multivariate Generalized Gaussian
        Distributions. IEEE trans on SP 2017.

        Args:
          params: Dictionary with required parameters for the sampler. Here this is
          the shape of a Gamma distribution and the dimension of the corresponding
          multivariate distribution. Key names should be 'shape' and 'dim'.

        Returns:
          sampler - a scalar sampler of a Gamma distribution.
        """
        beta = params['shape']
        p = params['dim']
        sampler = lambda m: np.power(np.random.gamma(p / (2 * beta), scale=2., size=(m, 1)),
                                     1. / (2 * beta))
        return sampler


class SyntheticDataset(Dataset):
    def __init__(self, p, Ns, M, beta, **kwargs):
        super().__init__(**kwargs)

        self.p = p
        self.Ns = Ns
        self.M = M
        self.beta = beta

        self.data = [[None for _ in range(M)] for _ in Ns]

        self._generate_data()

    def _generate_data(self):
        max_N = max(self.Ns)
        self.X = []
        self.Ks = []
        self.Es = []

        for j in range(self.M):
            K = generate_random_sparse_psd(self.p, 0.6)
            E = generate_inverse_covariance_structure(K)

            self.Ks.append(K)
            self.Es.append(E)
            self.X.append(self._sample(K, max_N))

            for i, N in enumerate(self.Ns):
                self.data[i][j] = self.X[j][:, :N]
                #print(np.linalg.norm(K - np.linalg.pinv(util.sample_covariance(self.data[i][j])), 'fro'))

        #exit()

    def _sample(self, K, N):
        #return multivariate_generalized_gaussian(K, self.beta, self.p, N)
        #return get_elliptic_data(PositiveScalarSamplerFactory().multivariate_t({'dim': self.p, 'nu': 2}), self.p, N, K)
        return get_elliptic_data(PositiveScalarSamplerFactory().generalized_gaussian({'dim': self.p, 'shape': self.beta}), self.p, N, K)

    def get_dimension(self):
        return self.p

    def get_Ns(self):
        return self.Ns

    def get_averaging(self):
        return self.M

    def get_edges(self, N_index, iteration):
        return self.Es[iteration]

    def get_train_set(self, N_index, iteration):
        return self.data[N_index][iteration]

    def get_test_set(self, N_index, iteration):
        raise NotImplementedError('This dataset does not have train/test splits')


class JointEstimationDistanceErrorMetric(ErrorMetric):
    def __init__(self, T, dataset=None, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset
        self.T = T

    def default_name(self):
        return 'Distance'

    def calculate(self, estimator_index, estimator, N_index, N, i):
        dataset = self.dataset

        if estimator.initial_N() is not None and N < estimator.initial_N():
            return np.nan

        if estimator.dataset() is not None:
            dataset = estimator.dataset()

        X = dataset.get_train_set(N_index, i)

        K_hat = estimator.estimate_joint(X, dataset.get_edges(N_index, i), self.T)
        K_star = dataset.Ks[i]

        p = X.shape[0]
        value = np.trace(K_star) / np.trace(K_hat) * K_hat
        #value = p / np.trace(K_hat) * K_hat
        #ref = p / np.trace(K_star) * K_star
        ref = K_star
        #value, ref = K_hat, K_star
        error = np.linalg.norm(value - ref, 'fro') ** 2 / np.linalg.norm(ref, 'fro') ** 2

        return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dimension', type=int, default=DIMENSION_DEFAULT)
    parser.add_argument('-N', '--num-samples', type=int, default=[60, 70, 80, 100, 150, 250, 500, 850], nargs='+')
    parser.add_argument('-M', '--average-iterations', type=int, default=AVERAGE_ITERATIONS_DEFAULT)
    parser.add_argument('-b', '--beta', type=float, default=[1.0, 0.5, 0.2], nargs='+')
    args = parser.parse_args()

    p = args.dimension
    Ns = args.num_samples
    M = args.average_iterations
    T = 200

    TYLER_MAX_ITERS = 1000
    TYLER_NEWTON_STEPS = 750
    GAUSSIAN_NEWTON_STEPS = 750

    for beta in args.beta:
        dataset = SyntheticDataset(p=p, Ns=Ns, M=M, beta=beta)
        metric = JointEstimationDistanceErrorMetric(T=T, dataset=dataset)

        estimator_objects = [
            #estimators.joint.general_loss.MMNewtonJointEstimator(
            #   loss=losses.tyler(dataset.get_dimension()), tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS,
            #   newton_tol=1e-6, name='Tyler'),
            # estimators.joint.gauss_loss.NewtonJointEstimator(
            #     newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6, name='GMRF Newton'),
            # estimators.joint.general_loss.MMJointEstimator(
            #     estimators.joint.gauss_loss.InvestJointEstimator(), loss=losses.tyler(dataset.get_dimension()),
            #     tolerance=1e-6, max_iters=TYLER_MAX_ITERS, name='Tyler'),
            #estimators.joint.gauss_loss.InvestJointEstimator(name='GMRF'),
            estimators.joint.general_loss.MMNewtonJointEstimator(
               loss=losses.generalized_gaussian(beta, 1), tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS,
               newton_tol=1e-6, name='GG'),
            estimators.joint.gauss_loss.SampleCovarianceJointEstimator(name='Sample covariance')
        ]

        plots.plot_variables_vs_N(estimator_objects, Ns, M, metric, show=False)

    plots.show()


if __name__ == '__main__':
    main()
