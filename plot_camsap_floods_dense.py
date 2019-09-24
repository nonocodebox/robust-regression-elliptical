import plots
import losses
import estimators
import regressors
from datasets import FloodsDataset
import argparse


NUM_SAMPLES_DEFAULT = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
AVERAGE_ITERATIONS_DEFAULT = 10

TYLER_MAX_ITERS = 12
TYLER_NEWTON_STEPS = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-samples', type=int, nargs='+', default=NUM_SAMPLES_DEFAULT)
    parser.add_argument('-M', '--average-iterations', type=int, default=AVERAGE_ITERATIONS_DEFAULT)
    args = parser.parse_args()

    T = 20
    M = args.average_iterations
    Ns = args.num_samples

    dataset = FloodsDataset(Ns=Ns, M=10, test_size=200, mode=FloodsDataset.Mode.DP_TO_D)
    dataset = dataset.structured_full()

    plots.plot_variables_vs_N(
        [
            regressors.common.LinearRegressor(name='Linear'),
            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=losses.tyler(dataset.get_dimension_y()),
                tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='ROMER-Tyler'),
            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=losses.generalized_gaussian(0.8, 1),
                tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='ROMER-GG m=1 beta=0.8'),
            regressors.conditional.ConditionalRegressor(estimators.conditional.general_loss.MMNewtonConditionalEstimator(
                loss=losses.generalized_gaussian(0.5, 1),
                tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS, newton_tol=1e-6
            ), name='ROMER-GG m=1 beta=0.5')
        ],
        dataset.get_Ns(),
        dataset.get_averaging(),
        plots.metrics.ConditionalRegressionNMSEErrorMetric(T, dataset=dataset, output_path='results-camsap-floods-dense.pickle'),
        independent_variable='Training set size'
    )


if __name__ == '__main__':
    main()
