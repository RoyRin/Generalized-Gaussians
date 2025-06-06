import numpy as np
import matplotlib.pyplot as plt
from data_aware_dp import rdp, beta_exponential


def plot_rdp_scaling(betas = None, x_range=[-4, 4], y_max = 25):
    x = np.linspace(x_range[0], x_range[1], 1000)
    if betas is None:
        beta_range=[.1, 4]
        betas = np.arange(beta_range[0], beta_range[1], .4)
    verbose = False

    plt.figure(figsize=(10, 10))

    plt.title(fr"R$\'enyi$ Diverence Scaling by $\alpha$")
    plt.ylabel(r"R$\'e$nyi divergence")
    plt.xlabel(r"$\alpha$")
    alphas = np.arange(2, 100, 0.3)
    if False:
        p1 = beta_exponential.beta_exponent(x, beta=2, shift=0)
        p1 = p1 / sum(p1)  # rescale
        p2 = beta_exponential.beta_exponent(x, beta=2,
                                            shift=1)  # assume sensitivity = 1
        p2 = p2 / sum(p2)
        renyi_divergence_values = [
            rdp.renyi_divergence(p1, p2, alpha_i) for alpha_i in alphas
        ]
        renyi_divergence_values = [val[0] for val in renyi_divergence_values]
        plt.plot(alphas,
                 renyi_divergence_values,
                 label=f"beta = {np.around(beta,2)}",
                 color="red",
                 linewidth=2)

    for beta in betas:
        p1 = beta_exponential.beta_exponent(x, beta=beta, shift=0)
        p1 = p1 / sum(p1)  # rescale
        p2 = beta_exponential.beta_exponent(x, beta=beta,
                                            shift=1)  # assume sensitivity = 1
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
    verbose = False

    plt.figure(figsize=(10, 10))

    plt.title(
        f"Removed points RDP scaling for alpha, given (exp(-|x|^beta) mechanism"
    )
    plt.ylabel("Removed points")
    plt.xlabel("alpha")
    alphas = np.arange(2, 100, 0.3)

    for beta in betas:
        p1 = beta_exponential.beta_exponent(x, beta=beta, shift=0)
        p1 = p1 / sum(p1)  # rescale
        p2 = beta_exponential.beta_exponent(x, beta=beta,
                                            shift=1)  # assume sensitivity = 1
        p2 = p2 / sum(p2)

        renyi_divergence_values = [
            rdp.renyi_divergence(p1, p2, alpha_i) for alpha_i in alphas
        ]
        removed_pts = [val[1] for val in renyi_divergence_values]
        renyi_divergence_values = [val[0] for val in renyi_divergence_values]
        plt.plot(alphas, removed_pts, label=f"beta = {np.around(beta,2)}")

    plt.legend()
    plt.show()