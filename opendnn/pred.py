import theano.tensor

def resolve(pred_fn):
    if not pred_fn in globals():
        raise NameError("Unknown prediction function: {}".format(pred_fn))

    return globals().get(pred_fn)

def argmax(y_hat):
    return theano.tensor.argmax(y_hat, axis=1)
