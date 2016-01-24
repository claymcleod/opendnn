import numpy

def uniform(size, rng):
    return numpy.random.uniform(low=-rng, high=rng, size=size)

def normal(size, rng):
    return numpy.random.normal(loc=0.0, scale=rng, size=size)

def lecun(dim_in, dim_out, use_normal=False):
    if use_normal:
        return normal((dim_in, dim_out), numpy.sqrt(1. / dim_in))
    else:
        return uniform((dim_in, dim_out), numpy.sqrt(3. / dim_in))

def glorot(dim_in, dim_out, use_normal=False):
    if use_normal:
        return normal((dim_in, dim_out), numpy.sqrt(3. / (dim_in + dim_out)))
    else:
        return uniform((dim_in, dim_out), numpy.sqrt(6. / (dim_in + dim_out)))

def he(dim_in, dim_out, use_normal=False):
    if use_normal:
        return normal((dim_in, dim_out), numpy.sqrt(2. / dim_in))
    else:
        return uniform((dim_in, dim_out), numpy.sqrt(6. / dim_in))
