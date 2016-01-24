import theano.tensor

def resolve(loss_fn):
    if not loss_fn in globals():
        raise NameError("Unknown loss function: {}".format(loss_fn))

    return globals().get(loss_fn)

def categorical_crossentropy(y, y_hat):
    return theano.tensor.nnet.categorical_crossentropy(y, y_hat).mean()

def mean_squared_error(y, y_hat):
    return theano.tensor.mean((y - y_hat) ** 2)
