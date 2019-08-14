import plots
import losses
import estimators
import regressors
from datasets import FloodsDataset
import argparse


AVERAGE_ITERATIONS_DEFAULT = 10
NUM_SAMPLES_DEFAULT = list(range(30, 401, 10))


def main():
    #M = 5
    T = 50

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-samples', type=int, nargs='+', default=NUM_SAMPLES_DEFAULT)
    parser.add_argument('-M', '--average-iterations', type=int, default=AVERAGE_ITERATIONS_DEFAULT)
    args = parser.parse_args()

    M = args.average_iterations
    Ns = args.num_samples
    #Ns = list(range(30, 401, 10))
    #Ns = list(range(30, 401, 20))

    dataset = FloodsDataset(Ns=Ns, M=M, test_size=200, mode=FloodsDataset.Mode.DP_TO_D)
    dataset_full = dataset.structured_full()
    dataset_time = dataset.structured_time()
    dataset_timespace = dataset.structured_timespace()

    loss = losses.tyler(dataset.get_dimension())

    TYLER_MAX_ITERS = 12
    TYLER_NEWTON_STEPS = 25
    GAUSSIAN_NEWTON_STEPS = 200

    dx = dataset.get_dimension_x()
    dy = dataset.get_dimension_y()

    plots.plot_variables_vs_N(
        [
            # regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(
            #     loss=loss, tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS,
            #     newton_tol=1e-6
            # ), name='Tyler time', dataset=dataset_time, initial_N=30),
            # regressors.joint.JointRegressor(dx, dy, estimators.joint.gauss_loss.NewtonJointEstimator(
            #     newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            # ), name='GMRF time', dataset=dataset_time, initial_N=30),
            #
            # regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(
            #     loss=loss, tolerance=1e-6, max_iters=TYLER_MAX_ITERS, newton_num_steps=TYLER_NEWTON_STEPS,
            #     newton_tol=1e-6
            # ), name='Tyler time-space', dataset=dataset_timespace, initial_N=60),
            # regressors.joint.JointRegressor(dx, dy, estimators.joint.gauss_loss.NewtonJointEstimator(
            #     newton_num_steps=GAUSSIAN_NEWTON_STEPS, newton_tol=1e-6
            # ), name='GMRF time-space', dataset=dataset_timespace, initial_N=50),

            regressors.joint.JointRegressor(dx, dy, estimators.joint.log_loss.TylerJointEstimator(),
                                            name='Tyler', dataset=dataset_full, initial_N=180),
            regressors.common.LinearRegressor(dx=dx, dy=dy, name='GMRF', dataset=dataset_full, initial_N=240),
            regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(losses.generalized_gaussian(0.5, 1)),
                                            name='GG 0.5', dataset=dataset_full, initial_N=180),
            regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(losses.generalized_gaussian(0.2, 1)),
                                            name='GG 0.2', dataset=dataset_full, initial_N=180),
            regressors.joint.JointRegressor(dx, dy, estimators.joint.general_loss.MMNewtonJointEstimator(losses.multivariate_t(dx+dy, 4)),
                                            name='T 2.5', dataset=dataset_full, initial_N=180),

        ],
        dataset.get_Ns(),
        dataset.get_averaging(),
        plots.metrics.JointRegressionNMSEErrorMetric(T),
        independent_variable='Training set size'
    )


if __name__ == '__main__':
    main()
