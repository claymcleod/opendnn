import theano.tensor


def categorical_crossentropy(y, y_hat):
    return theano.tensor.nnet.categorical_crossentropy(y, y_hat).mean()


def mean_squared_error(y, y_hat):
    return theano.tensor.mean((y - y_hat) ** 2)
