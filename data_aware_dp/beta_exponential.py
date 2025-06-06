import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy

JAX_INSTALLED = True
try:
    from jax import numpy as jnp
except ImportError:
    JAX_INSTALLED = False

# coef = beta_ / (2 * np.power(scale,1/ beta) * scipy.special.gamma( 1. / beta_))

def jax_installed_decorator(f):
    if JAX_INSTALLED:
        return f

    return lambda: print("Jax not installed")


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
    coef = beta / (2 * np.power(scale,1/ beta) * scipy.special.gamma( 1. / beta))
    return coef

    inv = 2 * scipy.special.gamma(1 / beta)
    return beta * c / inv


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


def beta_exponent__torch(x, beta, c, k=None, shift=0.):
    """ same as beta_exponent but for torch
    """
    if k is None:
        k = solve_for_k(beta, c)
    return k * torch.exp(-1. * c * ((torch.abs(x - shift))**beta))


@jax_installed_decorator
def beta_exponent__jax(x, beta, c, k=None, shift=0.):
    """ same as beta_exponent but for jax
    """
    if k is None:
        k = solve_for_k(beta, c)
    return k * jnp.exp(-1. * c * ((jnp.abs(x - shift))**beta))


def gaussian(x, c=1):
    return beta_exponent(x, beta=2, c=c)


def laplacian(x, c=1):
    return beta_exponent(x, beta=1, c=c)


def plot_beta_exponent(beta,
                       c,
                       k=None,
                       shift=0,
                       pt_count=5000,
                       xrange=None,
                       show=True):
    """plot beta exponential

    Returns:
        _type_: _description_
    """

    if xrange is None:
        tol = 1e-3

        xmax = solve_for_tail(beta, c, k=k, tol=tol)
        xrange = [-1 * xmax, xmax]

    x = np.linspace(xrange[0], xrange[1], pt_count)
    p1 = beta_exponent(x, beta=beta, c=c, k=k, shift=shift)

    if show:
        plt.plot(x, p1, label=fr"$\beta$ ={beta}; c={np.round(c,2)}")
    return x, p1


def __beta_exponential_CDF(x, beta, c, shift=0):
    """ compute the CDF of the beta-exponential distribution"""
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammainc.html
    gamma = scipy.special.gammainc(1 / beta,
                                   np.power(c * np.abs(x - shift), beta))
    term = np.sign(x - shift) / 2
    term2 = 1 / (2 * scipy.special.gamma(1 / beta))
    ret = 1 / 2 + (term * gamma * term2)
    return ret


def beta_exponential_derivative_f(x, beta, c, k=1):
    """returns the derivative of the beta-exponential distribution, evaluated at point x"""
    constant = -1 * k * beta * c * (x * np.abs(x)**(beta - 1))
    return constant * np.exp(-1. * c * np.abs(x)**beta)


def invert_beta_exponential(y, beta, c, k):
    """given a value y, find the x such that beta_exponential(x) = y
        only retuns positive x

    Args:
        y (_type_): _description_
        beta (_type_): _description_
        c (_type_): _description_
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # y = k * np.exp(-1. * c * ((np.abs(x - shift))**beta))
    # np.log(y/k) = -1. * c * ((np.abs(x - shift))**beta)
    # np.log(y/k) / -1. * c = ((np.abs(x - shift))**beta)
    # (np.log(y/k) / -1. * c)**(1/beta) = (np.abs(x - shift))
    y = np.float128(y)
    if k is None:
        k = solve_for_k(beta, c)
    k = np.float128(k)
    c = np.float128(c)
    beta = np.float128(beta)

    log = np.log(y / k)
    log = log / (-1. * c)
    ret = np.power(log, 1 / beta)
    if np.isnan(ret):
        raise ValueError("y is too small")
    return float(ret)


def solve_for_tail(beta, c, k=None, tol=1e-3):
    """solve for a value x, that is at least as large at a value that gets f(x) = tol"""

    return invert_beta_exponential(tol, beta, c, k)
