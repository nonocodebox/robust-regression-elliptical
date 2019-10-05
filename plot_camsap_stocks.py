import numpy as np
import os
import os.path
from datetime import datetime, date
from csv import DictReader
import pickle
import argparse
from sklearn.covariance import GraphLasso, GraphLassoCV
import regressors
import estimators
import plots
import losses
from plots.metrics import ErrorMetric
from datasets import LabeledDataset
from util.data import generate_inverse_covariance_structure
from util import split_edges
import time


STOCKS_DIR = 'Data/Stocks/'


class StocksDataset(LabeledDataset):
    def __init__(self, Ns, divisions, shuffling, num_observed=275, num_hidden=67, **kwargs):
        super().__init__(**kwargs)

        self.Ns = Ns
        self.shuffling = shuffling
        self.divisions = divisions
        self.num_observed = num_observed
        self.num_hidden = num_hidden
        self.num_stocks = num_hidden + num_observed

        self._load_stocks_data()

    def get_dimension_x(self):
        return self.num_observed

    def get_dimension_y(self):
        return self.num_hidden

    def get_Ns(self):
        return self.Ns

    def get_averaging(self):
        return self.shuffling * self.divisions

    def get_edges(self, N_index, iteration):
        return self.Es[iteration]

    def get_train_set(self, N_index, iteration):
        return self.data[N_index][iteration]['train']

    def get_test_set(self, N_index, iteration):
        return self.data[N_index][iteration]['test']

    # For each stock return a dictionaty date->(open, close)
    def _read_ticker(self, filename):
        with open(os.path.join(STOCKS_DIR, filename)) as csvfile:
            reader = DictReader(csvfile)
            # TODO: Check duplicate dates?
            return dict([
            (
                datetime.strptime(row['Date'], '%Y-%m-%d').date(),
                (float(row['Open']), float(row['Close']))
            ) for row in reader])

    def _load_stocks_data(self):
        cache = 'camsap_stocks_cache.pickle'
        train_start_date = date(2004, 1, 1)
        crisis_start_date = date(2007, 7, 1)
        crisis_end_date = date(2009, 7, 1)
        train_end_date = date(2011, 6, 30)
        test_start_date = date(2011, 7, 1)
        test_end_date = date(2017, 10, 31)

        # Read from cache - preprocessed data.
        if os.path.isfile(cache):
            with open(cache, 'rb') as f:
                train_data, test_data, names = pickle.load(f)
        else:
            dicts = {}
            dates = []

            for filename in sorted(os.listdir(STOCKS_DIR)):
                # Stocks names are written in capital letters.
                name = filename.split('.')[0].upper()
                dict = self._read_ticker(filename)

                if len(dict) == 0:
                    print('Skipping ticker "{}": Empty'.format(name))
                    continue

                # Drop stocks by defined dates.
                first_date, last_date = min(dict.keys()), max(dict.keys())
                if first_date > train_start_date or last_date < test_end_date:
                    print('Skipping ticker "{}": First={}, Last={}'.format(name, first_date, last_date))
                    continue

                dicts[name] = dict
                dates += dict.keys()

            # Total dates appeared in stocks records.
            train_dates = set(d for d in set(dates) if train_start_date <= d <= train_end_date and (d < crisis_start_date or d > crisis_end_date))
            test_dates = set(d for d in set(dates) if test_start_date <= d <= test_end_date)

            print('Loaded total {} tickers for {} dates.'.format(len(dicts), len(train_dates) + len(test_dates)))

            names = []

            for name, dict in dicts.items():
                # Check that all stocks records have same dates (comparable).
                ticker_dates = set(dict.keys())
                train_diff = train_dates - ticker_dates
                test_diff = test_dates - ticker_dates
                if len(train_diff) != 0 or len(test_diff) != 0:
                    print('Error: Ticker "{}" has missing dates:'.format(name))
                    #print(', '.join([d.strftime('%Y-%m-%d') for d in train_diff.union(test_diff)]))
                    print('Skipping.')
                    continue

                names.append(name)

            train_data = []
            test_data = []

            for d in sorted(train_dates):
                train_data.append([dicts[name][d] for name in names])

            for d in sorted(test_dates):
                test_data.append([dicts[name][d] for name in names])

            with open(cache, 'wb') as f:
                pickle.dump((train_data, test_data, names), f)

        # dates x num_stocks x 2
        #  O   C
        #  O   C
        #   ...

        Z_train = np.array(train_data)
        Z_test = np.array(test_data)

        self.stock_values_train = (Z_train[:, :, 1] - Z_train[:, :, 0]) / Z_train[:, :, 0] # Intra-day returns
        self.stock_values_test = (Z_test[:, :, 1] - Z_test[:, :, 0]) / Z_test[:, :, 0]

        self.stock_names = names
        self.N_train = self.stock_values_train.shape[0]

        self._shuffle_data()

    def _generate_structure_K(self, X):
        lasso = GraphLassoCV(alphas=20)

        lasso.fit(X.T)
        K_structure = lasso.get_precision()

        if (hasattr(lasso, 'alpha_')):
            print('alpha=', lasso.alpha_)

        return K_structure

    def _shuffle_data(self):
        stocks_train = self.stock_values_train.T
        stocks_test = self.stock_values_test.T
        self.Es = []
        self.data = [
            [
                {'train': None, 'test': None, 'sigma': None} for _ in range(self.divisions * self.shuffling)
            ] for _ in self.Ns
        ]

        for k in range(self.divisions):
            np.random.seed(k)
            stock_order = np.random.permutation(stocks_train.shape[0])[:self.num_stocks]

            division_train = stocks_train[stock_order, :]
            division_test = stocks_test[stock_order, :]

            mu = np.mean(division_train, axis=1, keepdims=True)
            sigma = np.std(division_train, axis=1, keepdims=True)

            K_division = self._generate_structure_K((division_train - mu) / sigma)

            for j in range(self.shuffling):
                np.random.seed(1000 + j)
                train_indices = np.random.permutation(self.N_train)

                X_train_full = (division_train - mu) / sigma
                X_train = X_train_full[:, train_indices]
                X_test = (division_test - mu) / sigma

                K = K_division
                E = generate_inverse_covariance_structure(K)
                self.Es.append(E)

                for i, N in enumerate(self.Ns):
                    m = k * self.shuffling + j
                    self.data[i][m]['train'] = X_train[:, :N]
                    self.data[i][m]['test'] = X_test
                    self.data[i][m]['sigma'] = sigma


class StocksMSEErrorMetric(ErrorMetric):
    def __init__(self, T, divisions, dataset=None, **kwargs):
        super().__init__(**kwargs)

        self.dataset = dataset
        self.divisions = divisions
        self.T = T

    def default_name(self):
        return 'MSE'

    def calculate(self, estimator_index, regressor, N_index, N, i):
        XY_train = self.dataset.get_train_set(N_index, i)
        X_train = XY_train[:self.dataset.get_dimension_x(), :]
        Y_train = XY_train[self.dataset.get_dimension_x():, :]
        X_test = self.dataset.get_test_set_x(N_index, i)
        Y_test = self.dataset.get_test_set_y(N_index, i)
        E = self.dataset.get_edges(N_index, i)

        if isinstance(regressor, regressors.conditional.ConditionalRegressorBase):
            _, E_yx, E_yy = split_edges(E, self.dataset.get_dimension_x(), self.dataset.get_dimension_y())
            time_start = time.time()
            reg_coef = regressor.regress_conditional(X_train, Y_train, E_yx, E_yy, self.T)
            time_end = time.time()
        else:
            time_start = time.time()
            reg_coef = regressor.regress_joint(XY_train, E, self.T)
            time_end = time.time()

        duration = time_end - time_start

        Y_hat = reg_coef @ X_test

        error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2)# / np.mean(np.linalg.norm(Y_test, axis=0) ** 2)

        # pred_train_tyler = reg_coef.dot(X_train)
        # a = Y_train.dot(pred_train_tyler.T)
        # b = pred_train_tyler.dot(pred_train_tyler.T)
        # scaling_ls_tyler = np.trace(a) / np.trace(b)
        # reg = scaling_ls_tyler * reg_coef
        # # print('Y_test=', Y_test.shape)
        # # print('reg=', reg.dot(X_test).shape)
        # # print('sigma=', self.dataset.data[N_index][i]['sigma'].shape)
        # errs = (Y_test - reg.dot(X_test)) * self.dataset.data[N_index][i]['sigma'][self.dataset.get_dimension_x():, :]
        # error = np.mean(errs ** 2)

        #error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2 / np.linalg.norm(Y_test, axis=0) ** 2)
        print('mse=', error)
        return error, duration

    def postprocess(self, values):
        #print(values)
        #print('mean=')
        #print(np.min(values, axis=(0,1), keepdims=True))
        #print('final=')

        # Reshape to (Estimators x Ns x divisions x shuffling x metrics)
        sh = values.shape
        v = values.reshape(sh[0], sh[1], self.divisions, sh[2] // self.divisions, sh[3])

        with open('stock-results.pkl', 'wb') as f:
            pickle.dump(v, f)

        #v -= np.min(v, axis=(0, 1, 3), keepdims=True)
        v[:, :, :, :, 0] /= np.min(v[:, :, :, :, 0], axis=(0, 1, 3), keepdims=True)
        #print(v)

        # Reshape back to original shape
        v = v.reshape(sh)
        return v

    def metric_count(self):
        return 2

    def metric_name(self, index):
        if index == 0:
            return 'MSE'
        elif index == 1:
            return 'Runtime (seconds)'


def main():
    #M = 5
    T = 20#50

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--divisions', type=int, default=5)#default=75)
    parser.add_argument('-s', '--shuffling', type=int, default=5)#default=100)
    parser.add_argument('-N', '--num-samples', type=int, default=list(range(200, 901, 25)), nargs='+')
    args = parser.parse_args()

    #Ns = [200, 500, 800]
    Ns = args.num_samples

    dataset = StocksDataset(Ns=Ns, divisions=args.divisions, shuffling=args.shuffling, num_observed=105, num_hidden=15)

    #TYLER_MAX_ITERS = 12
    #TYLER_NEWTON_STEPS = 25
    TYLER_MAX_ITERS = 100
    TYLER_NEWTON_STEPS = 750
    GAUSSIAN_NEWTON_STEPS = 750

    dx = dataset.get_dimension_x()
    dy = dataset.get_dimension_y()

    plots.plot_variables_vs_N(
        [
            regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(
                loss=losses.tyler(dataset.get_dimension()), tolerance=1e-6, max_iters=TYLER_MAX_ITERS,
                newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='Tyler'),
            regressors.joint.JointRegressor(dx, dy, estimators.joint.gauss_loss.NewtonJointEstimator(
                newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            ), name='GMRF'),
            regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(
                loss=losses.generalized_gaussian_d(0.5, 1, dx + dy), tolerance=1e-6, max_iters=TYLER_MAX_ITERS,
                newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            ), name='Laplace'),

            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=losses.tyler(dataset.get_dimension()), tolerance=1e-6, max_iters=TYLER_MAX_ITERS,
                newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name = 'Tyler conditional'),
            regressors.conditional.ConditionalRegressor(estimators.conditional.gauss_loss.NewtonConditionalEstimator(
                newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            ), name='GCRF'),

            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=losses.generalized_gaussian_d(0.5, 1, dx + dy), tolerance=1e-6, max_iters=TYLER_MAX_ITERS,
                newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='Laplace conditional'),

            # regressors.joint.JointRegressor(dx, dy, estimators.joint.log_loss.TylerJointEstimator(),
            #                                 name='Unstructured Tyler'),

            # regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(
            #     loss=losses.generalized_gaussian(0.2, 1), tolerance=1e-6, max_iters=15,
            #     newton_num_steps=400, newton_tol=1e-6
            # ), name='GG m=1 b=0.2'),
            #regressors.joint.JointRegressor(dx, dy, estimators.joint.gauss_loss.InvestJointEstimator(), name='GMRF invest')
        ],
        dataset.get_Ns(),
        dataset.get_averaging(),
        StocksMSEErrorMetric(T, divisions=args.divisions, dataset=dataset),
        independent_variable='Training samples'
    )


if __name__ == '__main__':
    main()
