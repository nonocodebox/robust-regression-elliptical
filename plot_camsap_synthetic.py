import numpy as np
import plots
import regressors
import estimators
import losses
from datasets import LabeledDataset
import argparse
import statsmodels.sandbox.distributions.multivariate as mdist
from util.data import generate_inverse_covariance_structure, generate_random_sparse_psd


T_DIST_NU = 2.5
DIMENSION_X_DEFAULT = 3
DIMENSION_Y_DEFAULT = 7
AVERAGE_ITERATIONS_DEFAULT = 100
NUM_SAMPLES_DEFAULT = [20, 21, 22,23, 24, 25, 26, 27, 30, 33, 36, 39, 42, 45, 50, 70,
              80, 90, 100, 110, 120, 130, 140, 150, 170, 200, 250, 350, 500]


class SyntheticDataset(LabeledDataset):
    def __init__(self, dx, dy, Ns, M, dist='normal', **kwargs):
        super().__init__(**kwargs)

        self.dx = dx
        self.dy = dy
        self.Ns = Ns
        self.M = M
        self.dist = dist
        self.N_test = 500

        K_star = generate_random_sparse_psd(dx + dy, 0.6)

        self.K_star = K_star
        self.Q_star = np.linalg.pinv(self.K_star)

        self.E = generate_inverse_covariance_structure(self.K_star)
        #self.not_E = [(i, j) for i in range(self.p) for j in range(self.p) if (i, j) not in self.E]

        self.data = [[{'train': None, 'test': None} for _ in range(M)] for _ in Ns]

        self._generate_data()

    def _generate_data(self):
        max_N = max(self.Ns)
        self.X = []
        self.X_test = []

        for j in range(self.M):
            self.X.append(self._sample(max_N))
            self.X_test.append(self._sample(self.N_test))

            for i, N in enumerate(self.Ns):
                self.data[i][j]['train'] = self.X[j][:, :N]
                self.data[i][j]['test'] = self.X_test[j]

    def _sample(self, N):
        if self.dist == 'normal':
            return np.random.multivariate_normal(np.zeros(self.get_dimension()), self.Q_star, N).T
        elif self.dist == 't':
            return mdist.multivariate_t_rvs(np.zeros(self.get_dimension()), self.Q_star, df=T_DIST_NU, n=N).T
        else:
            raise Exception('Unknown distribution: ' + str(self.dist))

    def get_dimension_x(self):
        return self.dx

    def get_dimension_y(self):
        return self.dy

    def get_Ns(self):
        return self.Ns

    def get_averaging(self):
        return self.M

    def get_edges(self, N_index, iteration):
        return self.E

    def get_train_set(self, N_index, iteration):
        return self.data[N_index][iteration]['train']

    def get_test_set(self, N_index, iteration):
        return self.data[N_index][iteration]['test']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--distribution', choices=['normal', 't'], default='t')
    parser.add_argument('-x', '--dimension-x', type=int, default=DIMENSION_X_DEFAULT)
    parser.add_argument('-y', '--dimension-y', type=int, default=DIMENSION_Y_DEFAULT)
    parser.add_argument('-N', '--num-samples', type=int, default=NUM_SAMPLES_DEFAULT, nargs='+')
    parser.add_argument('-M', '--average-iterations', type=int, default=AVERAGE_ITERATIONS_DEFAULT)
    args = parser.parse_args()

    dx = args.dimension_x
    dy = args.dimension_y

    #Ns = [20, 21, 22,23, 24, 25, 26, 27, 30, 33, 36, 39, 42, 45, 50, 70,
    #      80, 90, 100, 110, 120, 130, 140, 150, 170, 200, 250, 350, 500]
    #Ns = [20, 30, 40, 100, 200]
    Ns = args.num_samples

    M = args.average_iterations
    T = 200

    dist = args.distribution

    dataset = SyntheticDataset(dx=dx, dy=dy, Ns=Ns, M=M, dist=dist)
    metric = plots.ConditionalRegressionNMSEErrorMetric(T=T, dataset=dataset)

    estimator_objects = [
        regressors.common.HuberRegressor(name='Huber'),
        regressors.conditional.ConditionalRegressor(estimators.conditional.gauss_loss.NewtonConditionalEstimator(newton_tol=1e-6), name='GCRF'),
        regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
            loss=losses.tyler(dy), tolerance=1e-6, max_iters=25, newton_tol=1e-6), name='ROMER-Tyler'),
        regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
            loss=losses.multivariate_t(dy, T_DIST_NU), tolerance=1e-6, max_iters=25, newton_tol=1e-6), name='ROMER-T-distribution')
    ]

    plots.plot_variables_vs_N(estimator_objects, Ns, M, metric)


if __name__ == '__main__':
    main()
