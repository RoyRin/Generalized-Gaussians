import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from opacus.accountants import utils as opacus_utils
from opacus.accountants.analysis import rdp as opacus_analysis_rdp

from data_aware_dp import utils as data_aware_dp_utils
from data_aware_dp import rdp as data_aware_dp_rdp
from opacus.accountants import create_accountant
import math

logger = logging.getLogger()
DEFAULT_RDP_ALPHA = 2
DEFAULT_Q_VALUE = 4096 / 45000

CWD = Path.cwd()

BASEDIR = CWD

mathematica_datadir = BASEDIR / "mathematica_simulations/"


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_SEPM_data(SEPM_filepath="SEPM_log_RDP_data.npz"):
    """ Loads the data from the SEPM RDP data file 
    
    """

    data = np.load(SEPM_filepath)
    RDP_data = data["arr_0"]
    betas = data["arr_1"]
    cvals = data["arr_2"]
    return RDP_data, betas, cvals


def integer_SEPM_q_calc(q, moments):
    """ Solve for the Sampled-Beta Mechanism 
    Assumes integer-valued alpha
    Args:
        q (float): q value (sample rate)
        moments [list[float]]: moments of the Sampled-beta mechanism
            Note: skip the 0th moment.
            
    
    (see 3.3 https://arxiv.org/pdf/1908.10530.pdf Renyi Differential Privacy of the Sampled Gaussian Mechanism ´)
    """
    alpha = len(moments)
    nChoosek = lambda n, k: scipy.special.binom(n, k)
    ret = 1. * ((1 - q)**alpha) * 1.
    #print(moments)
    for i, moment in enumerate(moments):
        k = i + 1  # ignoring the 0th momet
        if (moment == 0) or (moment == float("inf")):
            return None
        binom_coef = nChoosek(alpha, k)
        probability_coef = np.power((1 - q), (alpha - k)) * np.power(q, k)
        #print(f"binom_coef {binom_coef}\n probability_coef {probability_coef} moment {moment}")
        ret += binom_coef * probability_coef * moment
    if ret == 0:
        return None

    ret = np.log(ret) / (alpha - 1)
    if ret < 0:
        print(f"Return result {ret} < 0")
        return None
    return ret


def get_moments(beta, cval, betas, cvals, RDP_data):
    """ Returns the moments for a given beta and cval
    
    """
    beta_ind = np.where(betas == beta)[0][0]
    cval_ind = np.where(cvals == cval)[0][0]

    return [mat[beta_ind[beta], cval_ind[cval]] for mat in RDP_data]


def get_SEPM_beta_values():
    """
    get the list of beta values that have been precomputed 
    """
    simul_dir = mathematica_datadir  #/ "timing_rdp_calc_mistake/"
    path = simul_dir / "SEPM_log_RDP_data.npz"

    # RDP_data dimensions : [moment, beta, c] --> rdp-val
    RDP_data, betas, cvals = load_SEPM_data(SEPM_filepath=path)  # pass in path
    return betas


def get_log_moment_interpolators(beta):
    """
    returns a dictionary of functions that interpolates the log of the moment, for each power
        
    Returns: dictionary of RDP-interpolator function
        interpolators = { rdp_alpha_val: get_interpolator(rdp_alpha_val)    }
    """
    simul_dir = mathematica_datadir  #/ "timing_rdp_calc_mistake/"
    path = simul_dir / "SEPM_log_RDP_data.npz"

    # RDP_data dimensions : [moment, beta, c] --> rdp-val
    RDP_data, betas, cvals = load_SEPM_data(SEPM_filepath=path)  # pass in path
    if beta not in betas:
        raise Exception(f"beta {beta} not precomputed at {path}")

    moment_count = len(RDP_data)

    beta_ind = np.where(betas == beta)[0][0]

    # [moment, beta, c, rdp]
    def get_interpolator(values):
        """ Note: values is the log of the RDP values """
        interpolator = scipy.interpolate.interp1d(
            cvals,
            values,
            kind="cubic",
            #fill_value="extrapolate",
            bounds_error=True)
        return interpolator

    moment_interpolators = {}
    for moment_ind in range(moment_count):
        # note: shift the moment index by 1 because we're ignoring the 0th moment
        #  note - the interpolator returns the log of the value
        moment_interpolators[moment_ind + 1] = get_interpolator(
            RDP_data[moment_ind, beta_ind, :])

    return moment_interpolators


def c_to_gaussian_sigma(c):
    """ convert c to gaussian sigma

    beta: k * exp(-c *(x^beta)) # for beta =2 
    Gaussian: 1/sqrt(2pi) * exp(-x^2/(2 * sigma^2) )
    Args:
        c (float): c value
    """
    # c = 1/(2 * sigma^2)
    # sigma ^2 = 1/(2 * c)
    return np.sqrt(1 / (2 * c))


def gaussian_sigma_to_c(sigma):
    """ convert gaussian sigma to c
    Args:
        sigma (float): sigma value
    """
    # c = 1/(2 * sigma^2)
    return 1 / (2 * sigma**2)


def laplace_lambda_to_c(lambda_):
    """ convert laplace sigma to c
    Args:
        sigma (float): sigma value
    """
    return 1 / lambda_


def c_to_laplace_lambda(c):
    """ convert laplace sigma to c
    Args:
        sigma (float): sigma value
    """
    return 1 / c


def solve_for_optimal_c_val(target_epsilon,
                            epochs,
                            beta,
                            alphas=None,
                            sensitivity=1,
                            delta=1e-5,
                            tolerance=1e-3,
                            sample_rate=1):
    """Compute the appropriate c-value for a beta distribution, that would get the appropriate RDP value after N steps.

    Raises:
        Exception: _description_
    returns:
        c (float): c value for the beta distribution, that has the equivalent RDP value to the gaussian mechanism used.
    """

    log_SEPM_moment_interpolators = get_log_moment_interpolators(beta)
    # Note - this is a HACK - in that it only returns `c` when you pass in SEPM_interpolators. otherwise it returns sigma.
    #
    if alphas is None:
        alphas = list(log_SEPM_moment_interpolators.keys())

    c = opacus_utils.get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        steps=None,
        accountant="rdp",
        epsilon_tolerance=0.01,
        #
        alphas=alphas,
        log_SEPM_moment_interpolators=log_SEPM_moment_interpolators)
    return c


def solve_for_equivalent_c_val(original_beta,
                               original_c,
                               new_beta,
                               sample_rate,
                               steps=None,
                               epochs=None,
                               target_delta=1e-5):
    """ New: Pass in a beta= 2 and c values, for an expected number of steps
    compute an equivalent c value for a new beta value
    
    Note: will need to use scipy minimize 
    """
    #
    #
    if steps == None and epochs == None:
        raise Exception("Must pass either steps or epochs")

    original_log_SEPM_moment_interpolators = get_log_moment_interpolators(
        original_beta)
    new_log_SEPM_moment_interpolators = get_log_moment_interpolators(new_beta)

    alphas = original_log_SEPM_moment_interpolators.keys()
    # remove 0 and 1 from alphas
    print(alphas)
    alphas = [alpha for alpha in alphas if alpha not in [0, 1]]

    accountant = create_accountant(mechanism="rdp")
    #sigma = c_to_gaussian_sigma(original_c)
    accountant.history = [(original_c, sample_rate, steps)]
    # error occurs here:
    target_rdp, alpha = accountant.get_privacy_spent(
        delta=target_delta,
        #alphas=alphas,
        #
        log_SEPM_moment_interpolators=original_log_SEPM_moment_interpolators)

    print(target_rdp, alpha)

    target_eps, _ = opacus_analysis_rdp.get_privacy_spent(orders=alpha,
                                                          rdp=target_rdp,
                                                          delta=target_delta)

    def minimizer(c):
        accountant = create_accountant(mechanism="rdp")
        accountant.history = [(c, sample_rate, steps)]
        rdp_eps, alpha = accountant.get_privacy_spent(
            delta=target_delta,
            #alphas=alphas,
            #
            log_SEPM_moment_interpolators=new_log_SEPM_moment_interpolators)
        eps, _ = opacus_analysis_rdp.get_privacy_spent(orders=alpha,
                                                       rdp=rdp_eps,
                                                       delta=target_delta)
        return np.abs(float(eps) - target_eps)

    try:
        res = scipy.optimize.minimize_scalar(
            minimizer,
            bounds=(1e-11, 100),
            #x0=original_c,
            #method="Nelder-Mead",
            #tol=1e-3
        )
        return res.x  #[0]
    except Exception as e:
        print(e)
        return original_c
    #return minimizer


def ground_truth_rdp_laplace(c, mu=1, rdp_alpha=DEFAULT_RDP_ALPHA):
    """ compute the ground truth RDP for a laplace distribution

    NOTE: this only takes mu =1, because the proof has mu =1 hardcoded in the paper
    Args:
        sigma (float): sigma value
        alpha (float, optional): alpha value. Defaults to DEFAULT_RDP_ALPHA.

    Returns:
        float: RDP value
    """
    lambda_ = c_to_laplace_lambda(c=c)
    coef = 1 / (rdp_alpha - 1)
    term1 = ((rdp_alpha / ((2 * rdp_alpha) - 1)) * np.exp(
        (rdp_alpha - 1) / lambda_))
    term2 = (((rdp_alpha - 1) / ((2 * rdp_alpha) - 1)) *
             np.exp(rdp_alpha / lambda_))
    return coef * np.log(term1 + term2)


def ground_truth_rdp_gaussian(c, mu=1, rdp_alpha=DEFAULT_RDP_ALPHA):
    """ compute the ground truth RDP for a gaussian distribution

    Args:
        sigma (float): sigma value
        alpha (float, optional): alpha value. Defaults to DEFAULT_RDP_ALPHA.

    Returns:
        float: RDP value
    """
    sigma = c_to_gaussian_sigma(c=c)

    return rdp_alpha * (mu**2) / (2 * (sigma**2))


def rdp_to_epsilon(rdp_val, rdp_alpha, delta=1e-5):
    """ convert rdp to epsilon-delta 
    Args:
        rdp (float): renyi divergence
        alpha (float): alpha to specify renyi divergence alpha
    """
    epsilon = rdp_val + (np.log(1 / delta) / (rdp_alpha - 1))
    return (epsilon, delta)
