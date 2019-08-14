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
        cache = 'stocks-nocrisis.pickle'
        train_start_date = date(2004, 1, 1)
        train_end_date = date(2007, 7, 1)
        test_start_date = date(2009, 7, 1)
        test_end_date = date(2009, 12, 31)

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
            train_dates = set(d for d in set(dates) if d >= train_start_date and d <= train_end_date)
            test_dates = set(d for d in set(dates) if d >= test_start_date and d <= test_end_date)

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
            #names = list(dicts.keys())

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

        #Z = np.array(data)[:, :self.num_stocks, :]
        Z_train = np.array(train_data)[:, :self.num_stocks, :]
        Z_test = np.array(test_data)[:, :self.num_stocks, :]
        #Z -= np.mean(Z, axis=(0, 2), keepdims=True)

        self.stock_values_train = (Z_train[:, :, 1] - Z_train[:, :, 0]) / Z_train[:, :, 0] # Intra-day returns
        self.stock_values_test = (Z_test[:, :, 1] - Z_test[:, :, 0]) / Z_test[:, :, 0]

        mu = np.mean(self.stock_values_train, axis=0, keepdims=True)
        sigma = np.std(self.stock_values_train, axis=0, keepdims=True)
        self.stock_values_train -= mu
        self.stock_values_train /= sigma
        self.stock_values_test -= mu
        self.stock_values_test /= sigma

        # self.stock_values -= np.mean(self.stock_values, axis=0, keepdims=True)
        # self.stock_values /= np.std(self.stock_values, axis=0, keepdims=True)
        self.stock_names = names[:self.num_stocks]
        #self.N_test = int(self.stock_values.shape[0] * self.test_frac)
        #self.N_train = self.stock_values.shape[0] - self.N_test

        self.N_train = self.stock_values_train.shape[0]

        self._shuffle_data()

    def _generate_structure_K(self, X):
        # lasso = GraphLasso(alpha=0.012)
        lasso = GraphLassoCV(alphas=20)

        lasso.fit(X.T)
        K_structure = lasso.get_precision()

        if (hasattr(lasso, 'alpha_')):
            print('alpha=', lasso.alpha_)

        M = (np.abs(K_structure) > 1e-10)
        if (M == np.eye(M.shape[0], dtype=bool)).all():
            print('Got identity structure')
        # K_structure = np.ones(K_lasso.shape)

        return K_structure

    def _shuffle_data(self):
        stocks_train = self.stock_values_train.T
        stocks_test = self.stock_values_test.T
        self.Es = []
        self.data = [[{'train': None, 'test': None} for _ in range(self.divisions * self.shuffling)] for _ in self.Ns]

        # mu = np.mean(stocks[:, :self.N_train], axis=1, keepdims=True)
        # sigma = np.std(stocks[:, :self.N_train], axis=1, keepdims=True)
        # stocks -= mu
        # stocks /= sigma
        #stocks -= np.mean(stocks[:, :self.N_train], axis=1, keepdims=True)
        #stocks /= np.std(stocks[:, :self.N_train], axis=1, keepdims=True)

        K_structure = self._generate_structure_K(stocks_train)

        for k in range(self.divisions):
            np.random.seed(k)
            stock_order = np.random.permutation(self.num_stocks)

            for j in range(self.shuffling):
                np.random.seed(1000 + j)
                train_indices = np.random.permutation(self.N_train)

                X_train = stocks_train[stock_order, :][:, train_indices]
                X_test = stocks_test[stock_order, :]

                #K_lasso = self._generate_structure_K(X_train)
                #E = generate_inverse_covariance_structure(K_lasso)
                E = generate_inverse_covariance_structure(K_structure[stock_order, :][:, stock_order])
                self.Es.append(E)

                for i, N in enumerate(self.Ns):
                    m = k * self.shuffling + j
                    self.data[i][m]['train'] = X_train[:, :N]
                    self.data[i][m]['test'] = X_test


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
        X_test = self.dataset.get_test_set_x(N_index, i)
        Y_test = self.dataset.get_test_set_y(N_index, i)
        E = self.dataset.get_edges(N_index, i)

        reg_coef = regressor.regress_joint(XY_train, E, self.T)
        #print('coef=', reg_coef)

        Y_hat = reg_coef @ X_test
        #print('Y_hat=', Y_hat)

        error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2)# / np.mean(np.linalg.norm(Y_test, axis=0) ** 2)
        #error = np.mean(np.linalg.norm(Y_hat - Y_test, axis=0) ** 2 / np.linalg.norm(Y_test, axis=0) ** 2)
        print('mse=', error)
        return error

    def postprocess(self, values):
        #print(values)
        #print('mean=')
        #print(np.min(values, axis=(0,1), keepdims=True))
        #print('final=')

        # Reshape to (Estimators x Ns x divisions x shuffling)
        sh = values.shape
        v = values.reshape(sh[0], sh[1], self.divisions, sh[2] // self.divisions)

        with open('stock-results.pkl', 'wb') as f:
            pickle.dump(v, f)

        v -= np.min(v, axis=(0, 1, 3), keepdims=True)
        #print(v)

        # Reshape back to original shape
        v = v.reshape(sh)
        return v


def main():
    #M = 5
    T = 20#50

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--divisions', type=int, default=75)
    parser.add_argument('-s', '--shuffling', type=int, default=100)
    args = parser.parse_args()

    #Ns = list(range(200, 801, 50)) + [900]
    #Ns = [200, 300, 400, 600, 700, 900]
    Ns = [200, 500, 800]

    dataset = StocksDataset(Ns=Ns, divisions=args.divisions, shuffling=args.shuffling, num_observed=30, num_hidden=7)
    #dataset = StocksDataset(Ns=Ns, divisions=args.divisions, shuffling=args.shuffling, num_observed=60, num_hidden=15)
    #dataset = StocksDataset(Ns=Ns, divisions=args.divisions, shuffling=args.shuffling)

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
