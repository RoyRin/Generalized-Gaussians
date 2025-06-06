"""
python module, for evaluating beta-exponential mechanism 
"""
import datetime
import numpy as np
from data_aware_dp import sampling, histogram_computation
import pandas as pd

import multiprocessing
import itertools
import logging


def histogram_accuracy_evaluation(*,
                                  scale,
                                  beta,
                                  histograms,
                                  trials,
                                  verbose=False):
    """
    Analyze the accuracy of the beta-exponential mechanism for a given c and beta on a histogram


    Args:
        c_and_beta (_type_): _description_
        histograms (_type_): _description_
        alpha (_type_): _description_
        max_histogram_experiments (_type_): _description_
        trials (_type_): _description_

    Returns:
        pts (list): list of tuples (scaling of histogram groups, accuracy, beta, c, rdp)
    """
    start = datetime.datetime.now()
    sampler = sampling.beta_exponential_sampler_from_scale(beta=beta,
                                                           scale=scale)
    res = []
    for scaling, histogram_group in histograms.items():
        accs = histogram_computation.get_beta_accuracy(
            histograms=histogram_group, sampler=sampler, trials=trials)
        res.append((scaling, np.average(accs), np.std(accs), beta, scale))

    print(
        f"Time for c={scale}, beta={beta} : {datetime.datetime.now() - start}")
    return res


ARGS = None

TOTAL_COUNT = 0
NOW = datetime.datetime.now()
START = datetime.datetime.now()


def increment_logs():
    global TOTAL_COUNT, NOW, START

    if TOTAL_COUNT % 5 == 0:
        logging.debug(
            f"approximate progress : {(TOTAL_COUNT-1) * multiprocessing.cpu_count()}"
        )
        logging.debug(f"Time from start : {datetime.datetime.now() - START}")
        logging.debug(f"Time for last group : {datetime.datetime.now() - NOW}")
        NOW = datetime.datetime.now()

    TOTAL_COUNT += 1


def single_threaded_beta_computation(beta_scale_hists_trials):
    """ being changed right now"""
    #enumerated_c_and_beta_and_rdp):
    (beta, scale, histograms, trials) = beta_scale_hists_trials
    # global ARGS
    #count, c_and_beta = enumerated_c_and_beta
    # count, c_and_beta_and_rdp_val = enumerated_c_and_beta_and_rdp
    increment_logs()
    # c, beta, rdp_val = c_and_beta_and_rdp_val

    # histograms, rdp_alpha, max_histogram_experiments, trials = ARGS
    print(f"Computing for c={scale}, beta={beta}")
    try:
        return histogram_accuracy_evaluation(scale=scale,
                                             beta=beta,
                                             histograms=histograms,
                                             trials=trials,
                                             verbose=False)
    except Exception as e:
        logging.error(f"Error for scale={scale}, beta={beta}")
        logging.error(e)
        return []


def evaluate_beta_exponential_on_histograms(histograms,
                                            scales,
                                            betas,
                                            trials=60,
                                            savepath=None,
                                            max_cpus=None,
                                            verbose=True, should_multiprocessing = True):
    """evaluate_beta_exponential_on_histogram
        this function evaluates the beta-exponential mechanism on a histogram
        (the main function that calls all others)

    Args:
        histograms (dictionary): {entropy(int) : [np.array(histogram)]}
    returns:
        d, savepath : dataframe, savepath string
    """
    global ARGS, START
    START = datetime.datetime.now()
    logging.debug(f"Iterations: {len(scales)* len(betas)}")

    #cartesian_product = list(itertools.product(cs, betas))
    #enumerated_cartesian_product = [
    #    (i, c_and_beta) for i, c_and_beta in enumerate(cartesian_product)
    #]
    max_cpus = multiprocessing.cpu_count() if max_cpus is None else max_cpus

    # ARGS = (histograms, trials)

    beta_scales = itertools.product(betas, scales)
    beta_scales_hists_trials = [(beta, scale, histograms, trials)
                                for (beta, scale) in beta_scales]
    #
    # post process the rdp values.
    # rdp_vals = [rdp_calculator(c) for c in cs]
    # enumerate_beta_c_rdps = [(i, (c, beta, rdp_vals[i])) for i, c in enumerate(cs)]

    print(f"Total number to compute : {len(beta_scales_hists_trials)}")
    if should_multiprocessing:
        with multiprocessing.Pool(max_cpus) as p:
            pts = list(
                p.map(single_threaded_beta_computation,
                      beta_scales_hists_trials))

                # flatten the list 
        pts = list(itertools.chain(*pts))

    else:
        pts = list(
            map(single_threaded_beta_computation, beta_scales_hists_trials))

    # (scaling, np.average(np.array(accuracies)), beta, scale)
    d = pd.DataFrame(pts,
                     columns=[
                         "histogram_scaling", "accuracy", "accuracy_std",
                         "beta", "scale"
                     ])

    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    savepath = f"GGM_argmax_accuracy_{date_str}.csv" if not savepath else savepath
    d.to_csv(savepath)

    return d, savepath


#
# we want to find the curve, where for a given RDP value, the accuracy is the highest
#
def get_accuracies_for_rdp(df, rounding=0.1, rdp_range=[0, 10]):
    """ get the accuracy for a each RDP value
    returns dictionary of RDP value : top accuracy
    """
    get_nearest = lambda rdp_val: df.iloc[
        ((df["rdp"] - rdp_val).abs()).argsort()[:1]]  # .accuracy.iloc[0]

    #unique_rdps = np.array(list(set(np.around(df.rdp.unique(), rounding))))
    top_accuracies = {}

    for rdp_val in np.arange(start=rdp_range[0],
                             stop=rdp_range[1],
                             step=rounding):
        best_row = get_nearest(rdp_val)
        if best_row.empty:
            continue
        if np.abs(best_row.rdp.iloc[0] - rdp_val) > rounding:
            continue
        accuracy = best_row.accuracy.iloc[0]
        beta = best_row.beta.iloc[0]
        c = best_row.c.iloc[0]
        top_accuracies[rdp_val] = (accuracy, beta, c)
    return top_accuracies


def get_laplace_accuracy_at_rdp(rdp_val, laplacian_acc, tol=0.01):
    try:
        return laplacian_acc.loc[np.abs(
            laplacian_acc['rdp'] - rdp_val < tol)].iloc[0]["accuracy"]
    except:
        return None


def get_gaussian_accuracy_at_rdp(rdp_val, gaussian_acc, tol=0.01):
    try:
        return gaussian_acc.loc[np.abs(
            gaussian_acc['rdp'] - rdp_val < tol)].iloc[0]["accuracy"]
    except:
        return None
