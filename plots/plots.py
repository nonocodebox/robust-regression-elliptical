import warnings
import numpy as np
import matplotlib.pyplot as plt


def _nanmean(*args, **kwargs):
    """
    Wrapper for numpy's nanmean, ignoring any RuntimeWarnings. See numpy.nanmean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(*args, **kwargs)


def plot_variables_vs_N(estimators, Ns, M, error_metric, independent_variable='N', show=True):
    """
    Plots calculated variables as a function of an independent variable.
    :param estimators: List of EstimatorAlgorithm objects or estimator tag strings.
    :param Ns: List of Numpy array of N values to use.
    :param M: Number of averaging iterations.
    :param error_metric: An ErrorMetric instance for calculating the error metric.
    :param independent_variable: The name of the independent variable. Defaults to 'N'.
    """

    # Initialize a results array with NaNs
    values = np.full((len(estimators), len(Ns), M, error_metric.metric_count()), np.nan)

    for est_index, estimator in enumerate(estimators):
        for N_index, N in enumerate(Ns):
            for i in range(M):
                # Progress display
                progress = 100 * ((est_index * len(Ns) + N_index) * M + i + 1) / (len(estimators) * len(Ns) * M)
                print('Executing: {}, N={}, round={}/{} ({:.2f}%)'.format(
                    estimator.name(), N, i+1, M, progress), flush=True)

                # Metric calculation
                v = error_metric.calculate(est_index, estimator, N_index, N, i)
                context_text = 'Estimator={}, N={}, i={}'.format(estimators[est_index].name, N, i)

                if v is None:
                    print('({}) Error: Function did not return any value.'.format(context_text))
                    continue

                if len(v) < error_metric.metric_count():
                    print('({}) Error: Function did not return enough values.'.format(context_text))
                    continue

                # Store metric
                values[est_index, N_index, i, :] = v

    # Postprocess results
    values = error_metric.postprocess(values)

    for m in range(error_metric.metric_count()):
        # Plot results (mean over iterations)
        plt.figure()

        for est_index, estimator in enumerate(estimators):
            avg = _nanmean(values[est_index, :, :, m], axis=1)
            plt.plot(Ns, avg, label=estimator.name())

        variable_name = error_metric.metric_name(m)

        plt.title('{} vs. {}'.format(variable_name, independent_variable))
        plt.xlabel(independent_variable)
        plt.ylabel(variable_name)
        plt.legend()

    print('Done plotting')

    # Finally, show the plot if requested
    if show:
        plt.show()


def show():
    """
    Shows pending plots plotted with show=False.
    """
    plt.show()
