from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import integrate
from scipy.special import erfc

from ..rdp import _compute_rdp
from .domain import Domain
import scipy
from data_aware_dp import sampling

SQRT2 = np.sqrt(2)


def get_tail_bound(beta, scale, mu=1., tolerance=1e-3):
    """ Compute the value of x such that P(X > x) < tolerance
 
    exp(-(|x-mu|/scale)^beta) = tolerance
    """
    ret = np.power(-np.log(tolerance), 1 / beta)
    return ret * scale + mu


def scale_to_sigma(scale):
    return scale / np.sqrt(2)


def sigma_to_scale(sigma):
    return sigma * np.sqrt(2)


def cdf_prv_f__sampling(mu,
                        beta,
                        scale,
                        dimension=1,
                        N=100000,
                        bin_count=10000,
                        samples=None,
                        tolerance=1e-5):
    """ realization ! : if you set density = true, and then you aren't over the whole range, it wil renormalize such that the area under the curve is 1.


    """
    # #scale = sigma_to_scale(sigma)

    if samples is None:
        sampler = sampling.beta_exponential_sampler_from_scale(beta=beta,
                                                               scale=scale)
        samples = []
        if dimension != 1:
            for _ in range(N):
                samples.append(np.linalg.norm(sampler(dimension), ord=1))
            samples = np.array(samples)
        else:
            samples = sampler(N)

        Y = (np.power(np.abs(samples - mu), beta) -
             np.power(np.abs(samples), beta)) / (scale**beta)  #* -1
        X = Y * -1
        tail_bound = 2 * get_tail_bound(beta, scale, tolerance=tolerance)
        print(f"tail_bound is {tail_bound}")
        tail_bound = max(tail_bound, 70)
        tail_bound_left = -tail_bound
        tail_bound_right = tail_bound

    else:
        Y = samples
        X = samples * -1
        tail_bound_right = np.sign(max(samples)) * max(abs(samples)) * 1.1
        tail_bound_left = np.sign(min(samples)) * min(abs(samples)) * 0.9
        print(tail_bound_left)
        print(tail_bound_right)
        tail_bound_right = max(tail_bound_right, tail_bound_left)
        tail_bound_left = min(tail_bound_right, tail_bound_left)
    # print(tail_bound)
    # print(f"val at tail_bound is {np.exp(-tail_bound**beta / scale)}")

    bins = np.linspace(tail_bound_left, tail_bound_right, bin_count)

    x_hist, _ = np.histogram(X, bins=bins, density=True)
    y_hist, _ = np.histogram(Y, bins=bins, density=True)
    X_cdf = np.cumsum(x_hist) * (bins[1] - bins[0])
    Y_cdf = np.cumsum(y_hist) * (bins[1] - bins[0])
    # very slight averaging in X_cdf and Y _cdf, over window = 2
    window_ = 2
    X_cdf = np.convolve(X_cdf, np.ones(window_) / window_, mode='same')
    Y_cdf = np.convolve(Y_cdf, np.ones(window_) / window_, mode='same')

    print(f"interpolating")
    X_cdf_f_ = scipy.interpolate.interp1d(
        bins[:-1],
        X_cdf,
        kind='linear',  # 'quadratic',  #
        bounds_error=False,
        fill_value=(0, 1))

    def X_cdf_f(t):
        # cast t to float
        if isinstance(t, np.ndarray):
            return X_cdf_f_(t.astype(np.float32))
        return X_cdf_f_(float(t))

    Y_cdf_f_ = scipy.interpolate.interp1d(bins[:-1],
                                          Y_cdf,
                                          kind='linear',
                                          bounds_error=False,
                                          fill_value=(0, 1))

    def Y_cdf_f(t):

        # cast t to float
        if isinstance(t, np.ndarray):
            return Y_cdf_f_(t.astype(np.float32))
        return Y_cdf_f_(float(t))

    return X_cdf_f, Y_cdf_f


#
# EPD: exp(-(|x-mu|/scale)^beta)
# Gaussian: exp(- 1/2 (|x-mu|/sigma)^2)
# original Roy: exp(- c* (|x-mu|)^beta)
#
# Definitional: Sigma := Noise_multiplier
#

N = 10_000_000
bin_count = int(N / 50)


class PoissonSubsampledEPMPRV:
    r"""
    A Poisson subsampled Exponential Power Mechanism (EPM) privacy random variable.

    For details about the formulas for the pdf and cdf, see propositions B1 and B4 in
    https://www.microsoft.com/en-us/research/publication/numerical-composition-of-differential-privacy/

    Adding noise according to PRV for `exp(-(|x-mu|/scale)^beta)`
        where scale = sigma_to_scale(self.noise_multiplier)
    """

    def __init__(
        self,
        sample_rate: float,
        noise_multiplier: float,
        beta: float,
        mu=1.0,
        tol=1e-30,
        N=N,
        bin_count=bin_count,
        dimension=1,
    ) -> None:
        self.sample_rate = sample_rate
        #if noise_multiplier < .6:
        #    # HACK for now-  but if sigma is too small, then we have sampling errors
        #    raise Exception("noise_multiplier should be greater than 0.6")

        self.noise_multiplier = noise_multiplier
        self.beta = beta

        self.scale = sigma_to_scale(self.noise_multiplier)
        print("scale (from PoissonSubsampledEPMPRV)", self.scale)

        self.X_cdf_function, self.Y_cdf_function = cdf_prv_f__sampling(
            mu,
            beta,
            self.scale,
            N=N,
            bin_count=bin_count,
            tolerance=tol,
            dimension=dimension)

        self.mu = mu

    def pdf(self, t):
        raise Exception("Not implemented yet - Should not be called")

    def cdf(self, t):
        # Gaussian CDF for testing
        q = self.sample_rate
        z = np.log((np.exp(t) + q - 1) / q)
        if q == 1:
            return self.Y_cdf_function(z)

        return np.where(
            t > np.log(1 - q),
            (q * self.Y_cdf_function(z)) + ((1 - q) * self.X_cdf_function(z)),
            0.0,
        )


class PoissonSubsampledGenericPRV:
    r""" PoissonSubsampledGenericPRV """

    def __init__(
        self,
        cdf_f,  # call get_subsampled_Y_function__sampling
        sample_rate: float,
        noise_multiplier: float,
        beta: float,
        mu=1.0,
    ):
        # CDF should look something like : get_subsampled_Y_function__sampling(beta, scale, q, mu=1., N=100000000, bins=10000)
        self.cdf_f = cdf_f
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.beta = beta
        self.mu = mu

    def pdf(self, t):
        raise Exception("Not implemented yet - Should not be called")

    def cdf(self, t):
        return self.cdf_f(t)


class PoissonSubsampledGaussianPRV:
    r"""
    A Poisson subsampled Gaussian privacy random variable.

    For details about the formulas for the pdf and cdf, see propositions B1 and B4 in
    https://www.microsoft.com/en-us/research/publication/numerical-composition-of-differential-privacy/
    """

    def __init__(self, sample_rate: float, noise_multiplier: float) -> None:
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier

    def pdf(self, t):
        q = self.sample_rate
        sigma = self.noise_multiplier

        z = np.log((np.exp(t) + q - 1) / q)

        return np.where(
            t > np.log(1 - q),
            sigma * np.exp(-(sigma**2) * z**2 / 2 - 1 /
                           (8 * sigma**2) + 2 * t) /
            (SQRT2 * np.sqrt(np.pi) * (np.exp(t) + q - 1) *
             ((np.exp(t) + q - 1) / q)**0.5),
            0.0,
        )

    def cdf(self, t):
        q = self.sample_rate
        sigma = self.noise_multiplier

        z = np.log((np.exp(t) + q - 1) / q)

        return np.where(
            t > np.log(1 - q),
            -q * erfc((2 * z * sigma**2 - 1) / (2 * SQRT2 * sigma)) / 2 -
            (1 - q) * erfc(
                (2 * z * sigma**2 + 1) / (2 * SQRT2 * sigma)) / 2 + 1.0,
            0.0,
        )

    def rdp(self, alpha: float) -> float:
        return _compute_rdp(self.sample_rate, self.noise_multiplier, alpha)


# though we have only implemented the PoissonSubsampledGaussianPRV, this truncated prv
# class is generic, and would work with PRVs corresponding to different mechanisms
class TruncatedPrivacyRandomVariable:

    def __init__(self, prv: PoissonSubsampledGaussianPRV, t_min: float,
                 t_max: float) -> None:
        self._prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self._remaining_mass = self._prv.cdf(t_max) - self._prv.cdf(t_min)

    def pdf(self, t):
        return np.where(
            t < self.t_min,
            0.0,
            np.where(t < self.t_max,
                     self._prv.pdf(t) / self._remaining_mass, 0.0),
        )

    def cdf(self, t):
        return np.where(
            t < self.t_min,
            0.0,
            np.where(
                t < self.t_max,
                (self._prv.cdf(t) - self._prv.cdf(self.t_min)) /
                self._remaining_mass,
                1.0,
            ),
        )

    def mean(self) -> float:
        """
        Calculate the mean using numerical integration.
        """
        points = np.concatenate([
            [self.t_min],
            -np.logspace(-5, -1, 5)[::-1],
            np.logspace(-5, -1, 5),
            [self.t_max],
        ])

        mean = 0.0
        for left, right in zip(points[:-1], points[1:]):
            integral, _ = integrate.quad(self.cdf, left, right, limit=500)
            mean += right * self.cdf(right) - left * self.cdf(left) - integral

        return mean


@dataclass
class DiscretePRV:
    pmf: np.ndarray
    domain: Domain

    def __len__(self) -> int:
        if len(self.pmf) != self.domain.size:
            raise ValueError("pmf and domain must have the same length")
        return len(self.pmf)

    def compute_epsilon(self, delta: float, delta_error: float,
                        eps_error: float) -> Tuple[float, float, float]:
        if delta <= 0:
            return (float("inf"), ) * 3

        if np.finfo(
                np.longdouble).eps * self.domain.size > delta - delta_error:
            raise ValueError(
                "Floating point errors will dominate for such small values of delta. "
                "Increase delta or reduce domain size.")

        t = self.domain.ts
        p = self.pmf
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p * np.exp(-t)).cumsum())
        ndelta = np.exp(t) * d2 - d1

        def find_epsilon(delta_target):
            i = np.searchsorted(ndelta, -delta_target, side="left")
            if i <= 0:
                raise RuntimeError("Cannot compute epsilon")
            return np.log((d1[i] - delta_target) / d2[i])

        eps_upper = find_epsilon(delta - delta_error) + eps_error
        eps_lower = find_epsilon(delta + delta_error) - eps_error
        eps_estimate = find_epsilon(delta)
        return eps_lower, eps_estimate, eps_upper

    def compute_delta_estimate(self, eps: float) -> float:
        return np.where(
            self.domain.ts >= eps,
            self.pmf * (1.0 - np.exp(eps) * np.exp(-self.domain.ts)),
            0.0,
        ).sum()


def discretize(prv, domain: Domain) -> DiscretePRV:
    tC = domain.ts
    tL = tC - domain.dt / 2
    tR = tC + domain.dt / 2
    discrete_pmf = prv.cdf(tR) - prv.cdf(tL)

    mean_d = np.dot(domain.ts, discrete_pmf)
    # print(f"Discrete mean: {mean_d}")
    mean_c = prv.mean()
    #print(f"Continuous mean: {mean_c}")

    mean_shift = mean_c - mean_d
    #print(mean_shift)
    #print(domain.dt / 2)

    if np.abs(mean_shift) >= domain.dt / 2:
        raise RuntimeError(
            "Discrete mean differs significantly from continuous mean.")
    domain_shifted = domain.shift_right(mean_shift)

    return DiscretePRV(pmf=discrete_pmf, domain=domain_shifted)
