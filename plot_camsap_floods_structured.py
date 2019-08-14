import plots
import losses
import estimators
import regressors
from datasets import FloodsDataset
import argparse


NUM_SAMPLES_DEFAULT = [20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500]
AVERAGE_ITERATIONS_DEFAULT = 10


def main():
    T = 20

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-samples', type=int, nargs='+', default=NUM_SAMPLES_DEFAULT)
    parser.add_argument('-M', '--average-iterations', type=int, default=AVERAGE_ITERATIONS_DEFAULT)
    args = parser.parse_args()

    M = args.average_iterations
    Ns = args.num_samples
    #Ns = [60, 100]

    dataset = FloodsDataset(Ns=Ns, M=M, test_size=200, mode=FloodsDataset.Mode.DP_TO_D)
    dataset_full = dataset.structured_full()
    dataset_timespace = dataset.structured_timespace()

    loss = losses.tyler(dataset.get_dimension_y())

    TYLER_MAX_ITERS = 12
    TYLER_NEWTON_STEPS = 25
    GAUSSIAN_NEWTON_STEPS = 200

    plots.plot_variables_vs_N(
        [
            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=loss, tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='ROMER-Tyler time-space', dataset=dataset_timespace, initial_N=60),
            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=loss, tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='ROMER-Tyler', dataset=dataset_full, initial_N=100),
            regressors.conditional.ConditionalRegressor(estimators.conditional.gauss_loss.NewtonConditionalEstimator(
                newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            ), name='Gauss time-space', dataset=dataset_timespace, initial_N=60),
            regressors.common.LinearRegressor(name='Gauss', dataset=dataset_full, initial_N=175)
        ],
        dataset.get_Ns(),
        dataset.get_averaging(),
        plots.metrics.ConditionalRegressionNMSEErrorMetric(T),
        independent_variable='Training set size'
    )


if __name__ == '__main__':
    main()
