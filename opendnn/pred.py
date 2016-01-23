import theano.tensor


def argmax(y_hat):
    return theano.tensor.argmax(y_hat, axis=1)
