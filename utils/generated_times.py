import numpy as np

from utils.metrics import box_plots, hist_plots

seed = 31415
np.random.seed(seed)


def plot_predicted_distribution(predicted, empirical, data, time='days', cens=False):
    predicted_samples = np.transpose(predicted)
    print("observed_samples:{}, empirical_observed:{}".format(predicted_samples.shape,
                                                              empirical.shape))

    best_samples, diff, worst_samples = get_best_worst_indices(cens, empirical, predicted_samples)

    predicted_best = predicted_samples[best_samples]
    predicted_worst = predicted_samples[worst_samples]
    hist_plots(samples=diff, name='{}_absolute_error'.format(data), xlabel=r'|\tilde{t}-t|')

    box_plots(empirical=empirical[best_samples], predicted=list(predicted_best), name=('%s_best' % data),
              time=time)
    box_plots(empirical=empirical[worst_samples], predicted=list(predicted_worst), name=('%s_worst' % data),
              time=time)


def get_best_worst_indices(cens, empirical, predicted, size=50):
    diff = compute_relative_error(cens=cens, empirical=empirical, predicted=predicted)
    indices = sorted(range(len(abs(diff))), key=lambda k: diff[k])
    best_samples = indices[0:size]
    worst_samples = indices[len(indices) - size - 1: len(indices) - 1]
    return best_samples, diff, worst_samples


def compute_relative_error(cens, empirical, predicted, relative=False):
    predicted_median = np.median(predicted, axis=1)
    if cens:
        diff = np.minimum(0, predicted_median - empirical)
    else:
        diff = predicted_median - empirical
    if relative:
        return diff
    else:
        return abs(diff)
