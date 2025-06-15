import numpy as np
import matplotlib.pyplot as plt
from data_aware_dp import rdp
import scipy


def scale_from_c(c, beta):
    return 1 / (c**(1 / beta))


def c_from_scale(scale, beta):
    return (1 / scale)**beta


def solve_for_k(beta, c=1):  # k is the normalization constant
    """ numerically compute the normalization constant for the beta-exponential distribution
    so that it integrates to 1

        (k * exp(-c* |x|**beta)) 
    returns:
        k, tail_bound_error
    """
    scale = scale_from_c(c, beta)
    coef = beta / (2 * np.power(scale, 1 / beta) *
                   scipy.special.gamma(1. / beta))
    return coef


def beta_exponent(x, beta, c, k=None, shift=0.):
    """ Exponential Power Mechanism
    k * np.exp(-1. * c * ((np.abs(x - shift))**beta))

    Args:
        x (float or np.array): _description_
        beta (float) : 
        c (float): Defaults to 1.
        k (float): scaling constant. Defaults to 1.
        shift (int, optional): _description_. Defaults to 0.
    Returns:
        beta exponential distribution evaluated at x
    """
    if k is None:
        k = solve_for_k(beta, c)
    return k * np.exp(-1. * c * ((np.abs(x - shift))**beta))


def plot_rdp_scaling(betas=None, x_range=[-4, 4], y_max=25):
    x = np.linspace(x_range[0], x_range[1], 1000)
    if betas is None:
        beta_range = [.1, 4]
        betas = np.arange(beta_range[0], beta_range[1], .4)
    verbose = False

    plt.figure(figsize=(10, 10))

    plt.title(fr"R$\'enyi$ Diverence Scaling by $\alpha$")
    plt.ylabel(r"R$\'e$nyi divergence")
    plt.xlabel(r"$\alpha$")
    alphas = np.arange(2, 100, 0.3)

    for beta in betas:
        p1 = beta_exponent(x, beta=beta, shift=0)
        p1 = p1 / sum(p1)  # rescale
        p2 = beta_exponent(x, beta=beta, shift=1)  # assume sensitivity = 1
        p2 = p2 / sum(p2)

        renyi_divergence_values = [
            rdp.renyi_divergence(p1, p2, alpha_i) for alpha_i in alphas
        ]
        removed_pts = [val[1] for val in renyi_divergence_values]
        renyi_divergence_values = [val[0] for val in renyi_divergence_values]
        plt.plot(alphas,
                 renyi_divergence_values,
                 label=fr"$\beta$ = {np.around(beta,2)}")

        if verbose:
            print(f"for beta {beta}")
            print(np.around(renyi_divergence_values[:10], 2))
    plt.ylim(0, y_max)
    plt.xlim(0, 70)
    plt.legend()
    plt.show()


def plot_removed_points(beta_range=[1, 6], x_range=[-4, 4]):

    x = np.linspace(x_range[0], x_range[1], 1000)
    ## Plot the number of removed points
    betas = np.arange(beta_range[0], beta_range[1], .3)

    plt.figure(figsize=(10, 10))

    plt.title(
        f"Removed points RDP scaling for alpha, given (exp(-|x|^beta) mechanism"
    )
    plt.ylabel("Removed points")
    plt.xlabel("alpha")
    alphas = np.arange(2, 100, 0.3)

    for beta in betas:
        p1 = beta_exponent(x, beta=beta, shift=0)
        p1 = p1 / sum(p1)  # rescale
        p2 = beta_exponent(x, beta=beta, shift=1)  # assume sensitivity = 1
        p2 = p2 / sum(p2)

        renyi_divergence_values = [
            rdp.renyi_divergence(p1, p2, alpha_i) for alpha_i in alphas
        ]
        removed_pts = [val[1] for val in renyi_divergence_values]
        renyi_divergence_values = [val[0] for val in renyi_divergence_values]
        plt.plot(alphas, removed_pts, label=f"beta = {np.around(beta,2)}")

    plt.legend()
    plt.show()
