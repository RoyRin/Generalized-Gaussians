"""
module used for exploring RDP calculations

"""
import numpy as np
from data_aware_dp import beta_exponential

DEFAULT_RDP_ALPHA = 2
####################


def solve_for_target_rdp_val_for_gaussian_dp(
        target_epsilon,
        delta,
        sample_rate,  # batch_size / len(trainset)
        epochs,
        sensitivity,
        rdp_alpha=DEFAULT_RDP_ALPHA,
        verbose=False):  # = max_grad_norm = 0.1
    """
    Solve for the target RDP value for Gaussian DP
        (how much the RDP val is, for a gaussian that is seeking to acheive the same epsilon) )


    Gaussian DP:
        sigma = np.sqrt( 2 * np.log(1.25 / delta) * (sensitivity**2)/ epsilon**2  )
    therefore
    noise_multiplier = np.sqrt( 2 * np.log(1.25 / delta)/ epsilon**2  )
    epsilon = np.sqrt( 2 * np.log(1.25 / delta) ) /  noise_multiplier
    and
        sigma = noise_multiplier * (sensitivity)

    noise_multiplier: The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity 

    note: we add noise according to : std=self.noise_multiplier * self.max_grad_norm,


    Args:
        target_epsilon (_type_): _description_
        delta (_type_): _description_
        sample_rate (_type_): _description_
        sensitivity (_type_): _description_
    returns:
        target_rdp_val
    """
    noise_multiplier = utils.get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        steps=None,
        accountant="rdp",
        epsilon_tolerance=0.01)
    single_mechanism_epsilon = np.sqrt(
        2 * np.log(1.25 / delta)) / noise_multiplier
    if verbose:
        print(f"noise_multiplier - {noise_multiplier}")
        print(f"single_mechanism_epsilon - {single_mechanism_epsilon}")
    print(f"noise_multiplier - {noise_multiplier}")

    ###
    sigma = noise_multiplier * sensitivity
    # N(sigma) = k * exp(- 0.5 * (x)^2 / sigma^2)
    # beta(c) = k * exp(- c* (x)^beta)
    # c = .5 /sigma^2
    c = .5 / (sigma**2)
    print(f"gaussian c - {c}")
    gaussian_rdp_val = compute_rdp(beta=2, c=c, rdp_alpha=rdp_alpha)
    print(f"RDP value is {gaussian_rdp_val}")
    return gaussian_rdp_val


def compute_rdp(beta,
                c,
                rdp_alpha=DEFAULT_RDP_ALPHA,
                sensitivity=1,
                x_pt_count=100000,
                tolerance=1e-4):
    """compute the RDP value for a beta exponential

    Args:
        beta (_type_): _description_
        c (_type_): _description_
        rdp_alpha (_type_, optional): _description_. Defaults to DEFAULT_RDP_ALPHA.
        x_pt_count (int, optional): _description_. Defaults to 100000.

    Returns:
        _type_: _description_
    """
    beta, c, rdp_alpha = float(beta), float(c), float(rdp_alpha)
    k = beta_exponential.solve_for_k(beta=beta, c=c)

    tail_bound = beta_exponential.solve_for_tail(beta, c, k=k, tol=tolerance)

    tail_bound = max(tail_bound, 5) * 2
    #print(f"tail bound is {tail_bound}")
    x = np.linspace(-1. * tail_bound, tail_bound, x_pt_count)
    #vals = beta_eq(x) * beta_eq2(x)
    vals = f(x, c=c, beta=beta, alpha=rdp_alpha, mu=sensitivity)

    rdp_res = np.trapz(y=vals, x=x)
    #print(rdp_res)

    return np.log(rdp_res) / (rdp_alpha - 1)


def __renyi_divergence(p1, p2, x, rdp_alpha):
    """ 
    Deprecated because of issues

    numerical stable renyi-divergence calculation
    Args:
        p1 (array): array of probabilities
        p2 (array): array of probabilities
        alpha (float): alpha to specify renyi divergence alpha
    """
    summation = []
    rdp_alpha = np.float128(rdp_alpha)
    if len(p1) != len(p2):
        raise Exception("p1 and p2 must be the same length")

    renyi_div_computation = lambda i: np.float128(x[i]) * (np.float128(p1[
        i]) / np.float128(p2[i]))**(rdp_alpha)

    summation = [
        renyi_div_computation(i) for i in range(len(x))
        if p1[i] > 0 and p2[i] > 0
    ]
    # ignore the infinities
    summation = np.array(summation)[~np.isinf(summation)]
    removed_pts = len(p1) - len(summation)
    print(f"summation is : {summation}")
    renyi_div = 1 / (rdp_alpha - 1) * np.log(
        sum(summation)) if removed_pts == 0 else np.nan

    return renyi_div, removed_pts


def term1(x, beta, alpha):
    return alpha * (np.abs(x)**beta)


def term2(x, beta, alpha=DEFAULT_RDP_ALPHA, mu=1.):
    return (1 - alpha) * np.abs(x - mu)**beta


def f(x, c, beta, alpha=DEFAULT_RDP_ALPHA, mu=1.):
    return np.exp(-c * (term1(x, beta=beta, alpha=alpha) +
                        term2(x, beta=beta, alpha=alpha, mu=mu)))


def taylor_term(x, shift, beta, c, alpha, term):
    return np.power((-1 * c * alpha) *
                    (np.abs(x - shift)**beta), term) / math.factorial(term)


def taylor_series(x, beta, c, alpha, shift=0, terms=10):
    """ compute the taylor series for the RDP value 

    Args:
        x (_type_): _description_
        beta (_type_): _description_
        c (_type_): _description_
        alpha (_type_): _description_
        terms (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    taylors = [
        taylor_term(x=x, shift=shift, beta=beta, c=c, alpha=alpha, term=i)
        for i in range(terms)
    ]
    return sum(taylors)


def taylor_term_combined(x, shift, beta, c, alpha, term):
    return np.power((-1 * c * alpha) *
                    (np.abs(x - shift)**beta), term) / math.factorial(term)
