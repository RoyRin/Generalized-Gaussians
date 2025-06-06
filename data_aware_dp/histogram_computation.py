import scipy as sp

import numpy as np
from data_aware_dp import sampling


def generate_random_histogram(runner_up_gap_percent=20,
                              classes=10,
                              number_of_votes=1000):
    """
    Generate a random histogram given a runner up value

    Assumes that vote can only be 1 or 0 (sensitivty = 1)

    Args:
        runner_up_gap_percent (int, optional): _description_. Defaults to 20.
        classes (int, optional): _description_. Defaults to 10.
        number_of_votes (int, optional): _description_. Defaults to 100.
    returns :
        histogram (np.array): histogram of the random distribution
    """
    if classes == 2:
        a = 100
        b = a - runner_up_gap_percent
        histogram = np.array([a, b])
        histogram = histogram * number_of_votes / sum(histogram)
        return histogram

    # histogram will be : [a, a* r, a* r1*r2, ...<X>..]
    # where we fix E[X] = (n-3) *a* r1*r2 /2
    r = 1 - (runner_up_gap_percent / 100)
    step_down = 0.95  # the difference between 3rd largest and 2nd largest
    # (a + (a*r) + (a*r*stepdown) + E[X]  = number_of_votes
    helper_ = (1 + r + (r * step_down) + (r * step_down * (classes - 3) / 2))
    a = int(number_of_votes / helper_)

    hist = [a, int(a * r), int(a * r * step_down)]
    goal = number_of_votes - sum(hist)

    # fill in the remaining n-3 values from a uniform distribution,
    # such that sum of the remaining values = goal

    for _ in range(10000):
        remainder_group = np.random.randint(0,
                                            hist[-1],
                                            size=classes - len(hist))
        if sum(remainder_group) == goal:
            hist.extend(remainder_group)
            return np.array(hist)

    raise Exception("Could not generate histogram")


def entropy_calc(hist):
    """returns the entropy of a histogram"""
    return sum([-1 * hist[i] * np.log2(hist[i]) for i in range(len(hist))])


def normed_entropy_calc(hist):
    """returns the entropy of a histogram"""
    return sum([-1 * hist[i] * np.log2(hist[i])
                for i in range(len(hist))]) / len(hist)


# find random starting point
# do SGD, to measure the distance between the starting point and the entropy we want


def general_loss(hist, entropy_goal, sum_weighting=0.2):
    """ returns the loss function for the entropy
        minimize
            1. distance of the entropy of the histogram - entropy goal
            2. distance of the histogram - uniform distribution
    """
    entropy_dist = abs(normed_entropy_calc(hist) - entropy_goal)
    h = hist / sum(hist)
    sum_normalization = abs(sum(h) - 1.)
    return entropy_dist + sum_weighting * sum_normalization


# scipy optimization


def generate_random_histogram_with_given_entropy(entropy_goal,
                                                 classes=10,
                                                 tries=10,
                                                 tolerance=0.1):
    if entropy_goal < 2.6:

        cobyla = False
    else:
        cobyla = False  # True
    sum_weighting = 0.3 if not cobyla else 0.0

    loss_f = lambda x: general_loss(
        x, entropy_goal, sum_weighting=sum_weighting)
    min_loss = np.inf
    best_res = None
    for i in range(tries):
        start = np.random.random(size=classes)
        start = start / sum(start)  # normalize
        start[0] = i / tries
        start = start / sum(start)  # normalize

        if cobyla:
            res = sp.optimize.minimize(
                loss_f,
                start,
                method='COBYLA',
                constraints={
                    "type": "ineq",
                    "fun": lambda x: -1 * (abs(sum(x) - 1) - .05)
                }  # enforce sum to be 1 with a tolerance of .1
                ,
                tol=tolerance)
        else:

            bnds = [[0, 1]] * classes

            res = sp.optimize.minimize(loss_f,
                                       start,
                                       method='TNC',
                                       bounds=bnds,
                                       tol=tolerance)

        if loss_f(res.x) < min_loss:
            min_loss = loss_f(res.x)
            best_res = res
        if loss_f(res.x) < tolerance:
            return res

    if res.success:
        return res

    return best_res


def get_max_entropy(class_count):
    p = 1 / class_count
    return -1 * class_count * (p * np.log2(p))


def get_argmax(histogram):
    return np.argmax(histogram)


def add_beta_noise(histogram, beta):
    # beta_exponent(x, beta, shift = 0, sigma = 1)
    return histogram + np.random.exponential(scale=beta, size=len(histogram))


def add_noise_to_histograms(histograms, beta, c):
    """returns a list of noised histograms

    Args:
        histograms (list): list of np.histograms
        beta (_type_): beta value
        c (_type_): c scaling

    Returns:
        _type_: _description_
    """
    noised_histograms = []
    sample_beta_exponential = sampling.inverse_transform_sampling_beta_exponential(
        beta, c)

    #gen_inv_sample = sampling.sample_beta_distribution(beta, c)
    for histogram in histograms:
        randoms = np.random.random(size=len(histograms))
        noised_histogram = histogram + sample_beta_exponential(randoms)
        #noise = np.array([next(gen_inv_sample) for _ in range(len(histogram))])
        noised_histograms.append(noised_histogram)  #histogram + noise)
    return noised_histograms


def get_noised_argmax_accuracy(histogram, sampler, trials=100):
    """ given a histogram, get the accuracy of the argmax of the noisy histogram

    Args:
        histogram (_type_): _description_
        noise_source (_type_): _description_
        trials (_type_): _description_
        correct_label (_type_): _description_

    Returns:
        _type_: _description_
    """
    correct_label = get_argmax(histogram)
    correct = [
        int(get_argmax(histogram + sampler(len(histogram))) == correct_label)
        for _ in range(trials)
    ]
    return sum(correct) / trials


def get_beta_accuracy(histograms, sampler, trials=100):
    """ get the accuracy of the noised argmax, for a given (beta, c) value

    Args:
        histograms (list): list of np.histograms
        sampler (function takes u ~ U(0,1) and returns a noise value): ):     
        sampler = sampling.inverse_transform_sampling_beta_exponential(beta, c)
        
    Returns:
        _type_: _description_
    """
    return [
        get_noised_argmax_accuracy(histogram, sampler, trials=trials)
        for histogram in histograms
    ]
