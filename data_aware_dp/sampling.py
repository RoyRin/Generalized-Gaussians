import numpy as np
import torch
import scipy.interpolate
import scipy

JAX_INSTALLED = True
try:
    import jax
    from jax import numpy as jnp
    from jax import jit as jax_jit
except ImportError:
    JAX_INSTALLED = False


def jax_installed_decorator(f):
    if JAX_INSTALLED:
        return f
    return lambda: print("Jax not installed")


def scale_from_c(c, beta):
    return 1 / (c**(1 / beta))


# def c_from_scale(scale, beta):
#     return (1 / scale)**beta

# NOTE to self: is `scale` the same here as in accounting
# Observation - we should see a problem ,if increasing sigma improves the accuracy


def exponential_power_sampler(beta, scale, seed=0):
    """ EPD sampler - taken from https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_continuous_distns.py#L9398-L9482

    exp(-scale * |x|**beta)
    
    (note: see equation 9 from https://sci-hub.ru/https://doi.org/10.1080/00949650802290912)
    """
    # see [2]_ for the algorithm
    # see [3]_ for reference implementation in SAS

    a = 1 / beta
    b = np.power(scale, beta)
    rng = np.random.Generator(np.random.SFC64(seed))

    # make it faster

    def f(size=1.):
        """ Sample from the EPD distribution.
        exp(-scale * |x|**beta)
        
        # see equation 9 from https://sci-hub.ru/https://doi.org/10.1080/00949650802290912
        """
        z = rng.gamma(shape=a, scale=b, size=size)
        #z = np.random.gamma(1/beta, size=size)
        y = np.power(z, (1 / beta))
        # convert y to array to ensure masking support
        #y = np.asarray(y)
        mask = rng.random(size=y.shape) < 0.5
        y[mask] *= -1  #y[mask]
        return y

    return f


def beta_exponential_sampler(beta, c):
    """ 
    return sampler from beta exponential distribution (function can be called with no arguments or with size = # argument)
    
    exp(-scale * |x|**beta)
    """
    scale = scale_from_c(c=c, beta=beta)
    return exponential_power_sampler(beta=beta, scale=scale)
    #rv = scipy.stats.gennorm(beta=beta, scale=scale)  # returns numpy array
    #return rv.rvs


def beta_exponential_sampler_from_scale(beta, scale):
    """ 
    return sampler from beta exponential distribution (function can be called with no arguments or with size = # argument)

    exp(-scale * |x|**beta)
    """
    return exponential_power_sampler(beta, scale)
    #rv = scipy.stats.gennorm(beta=beta, scale=scale)  # returns numpy array
    #return rv.rvs


# used to be inverse_transform_sampling_beta_exponential__torch
def __beta_exponential_sampler__torch(beta, c, device=None):
    """ 
    return sampler function from beta exponential distribution (function can be called with no arguments or with size = # argument)

    """
    #device = torch.cuda()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = beta_exponential_sampler(beta, c)

    def torch_sampler(size=None):
        if size is None:
            return torch.from_numpy(sampler()).to(device)
        return torch.from_numpy(sampler(size=size)).to(device)

    return torch_sampler


def beta_exponential_sampler__torch(beta, scale, device=None, seed=0):
    """ EPD sampler - taken from https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_continuous_distns.py#L9398-L9482

    exp(-scale * |x|**beta)
    
    (note: see equation 9 from https://sci-hub.ru/https://doi.org/10.1080/00949650802290912)
    """
    # see [2]_ for the algorithm
    # see [3]_ for reference implementation in SASb
    beta, scale = float(beta), float(scale)
    print(beta)
    print(scale)
    print(type(beta))
    print(type(scale))

    #g_cpu = torch.Generator(device='cpu')
    #g_cuda = torch.Generator(device='cuda')

    #rng = np.random.Generator(np.random.SFC64(0))
    # beta = torch.Tensor.float(float(beta))
    #beta = torch.Tensor.float(beta)
    beta = torch.Tensor([beta]).float()
    scale = torch.Tensor([scale]).float()

    # scale = torch.Tensor.float(float(scale))
    # scale = torch.Tensor.float(scale)
    gamma_shape = 1 / beta
    # dist = torch.distributions.gamma.Gamma(gamma_shape,
    #                                        gamma_rate,
    #                                        device=device)

    gamma_scale = torch.pow(scale, beta)
    gamma_rate = 1 / gamma_scale

    beta = beta.to(device)
    scale = scale.to(device)
    dist = torch.distributions.gamma.Gamma(gamma_shape.to(device),
                                           gamma_rate.to(device))

    # make it faster
    def f(shape=1.):
        """ Sample from the EPD distribution.
        exp(-scale * |x|**beta)
        
        # see equation 9 from https://sci-hub.ru/https://doi.org/10.1080/00949650802290912
        """
        if isinstance(shape, int):
            shape = (shape, )

        z = dist.sample(shape)  # , generator=g_cpu
        #z = np.random.gamma(1/beta, size=size)
        y = torch.pow(z, (1 / beta))
        # convert y to array to ensure masking support
        mask = torch.rand(shape, device=device) < 0.5
        y[mask] *= -1  #y[mask]
        return y

    return f


@jax_installed_decorator
def beta_exponential_sampler__jax(beta, c, device=None):
    """ 
    return sampler function from beta exponential distribution (function can be called with no arguments or with size = # argument)

    """
    devices = jax.devices()
    if devices[0].platform != "cpu":
        print("using GPU")
    sampler = beta_exponential_sampler(beta, c)

    #@jax_jit
    def jax_sampler(shape=None):
        # note : This function will create arrays on JAX’s default device. For control of the device placement of data,
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html
        if shape is None:
            return jnp.array(sampler())
        randoms = jnp.array(sampler(size=np.prod(shape)))
        return randoms.reshape(shape)

    return jax_sampler


@jax_installed_decorator
def beta_exponential_sampler__jax(*, beta, c, device=None):
    """ 
    return sampler function from beta exponential distribution (function can be called with no arguments or with size = # argument)

    """
    devices = jax.devices()
    if devices[0].platform != "cpu":
        print("using GPU")
    sampler = beta_exponential_sampler(beta, c)

    #@jax_jit
    def jax_sampler(shape=None):
        # note : This function will create arrays on JAX’s default device. For control of the device placement of data,
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html
        if shape is None:
            return jnp.array(sampler())
        randoms = jnp.array(sampler(size=np.prod(shape)))
        return randoms.reshape(shape)

    return jax_sampler


@jax_installed_decorator
def beta_exponential_sampler_from_scale__jax(*, beta, scale, device=None):
    """ 
    return sampler function from beta exponential distribution (function can be called with no arguments or with size = # argument)

    """
    devices = jax.devices()
    if devices[0].platform != "cpu":
        print("using GPU")
    sampler = beta_exponential_sampler_from_scale(beta=beta, scale=scale)

    #@jax_jit
    def jax_sampler(shape=None):
        # note : This function will create arrays on JAX’s default device. For control of the device placement of data,
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html
        if shape is None:
            return jnp.array(sampler())
        randoms = jnp.array(sampler(size=np.prod(shape)))
        return randoms.reshape(shape)

    return jax_sampler
