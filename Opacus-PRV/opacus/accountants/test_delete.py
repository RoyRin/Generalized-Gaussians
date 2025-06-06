import numpy as np

seed = 1
rng = np.random.Generator(np.random.SFC64(seed))

def fast_sampler_1_arg( arg):
    """ Sample from the EPD distribution.
    exp(-scale * |x|**beta)
    
    # see equation 9 from https://sci-hub.ru/https://doi.org/10.1080/00949650802290912
    """
    scale, beta, shape = arg
    a = 1/ beta 
    b = np.power(scale, beta)
    z = rng.gamma(shape=a, scale=b, size=shape)
    #z = np.random.gamma(1/beta, size=size)
    y = np.power(z, (1 / beta))
    # convert y to array to ensure masking support
    #y = np.asarray(y)
    mask = rng.random(size=y.shape) < 0.5
    y[mask] *= -1  #y[mask]
    return y