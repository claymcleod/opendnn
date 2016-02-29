"""Module for experimental layers and other oddities"""

import theano
from opendnn.layers import Layer, Activation


class ActivationPool(Layer):

    def __init__(self, activation_fns, coefs=None, threshold=False,
                 trainable=True, *args, **kwargs):
        assert(len(activation_fns) > 0), ("Must have at least one activation function!")
        if coefs is None:
            coefs = [1. / len(activation_fns)] * len(activation_fns)

        assert(len(activation_fns) == len(coefs)), ("Must have the same number of "
                                                           "weights as activation_fns.")

        self.activation_fns = activation_fns
        self.coefs = coefs
        self.trainable = trainable
        self.threshold = threshold
        super(ActivationPool, self).__init__(*args, **kwargs)

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        self.activation_weights = []
        for x, activation_fn in enumerate(self.activation_fns):
            this_weight = theano.shared(self.coefs[x],
                                        name='AP{}_w{}'.format(layer_num, x))
            if self.threshold:
                X = theano.tensor.clip(this_weight, -self.coefs[x],
                                       self.coefs[x]) * activation_fn(X)
            else:
                X = this_weight * activation_fn(X)
            self.activation_weights.append(this_weight)
        return X

    def _get_updates_(self, loss, learning_rate):
        updates = []
        if not self.trainable: return updates
        for a in self.activation_weights:
            updates.append((a, a - theano.tensor.grad(loss, a) * learning_rate))
        return updates

class MReLU(ActivationPool):
    """Essentially a PD controller intuition using the ReLU function"""

    def __init__(self, *args, **kwargs):
        super(MReLU, self).__init__([
            Activation('relu').nonlinearity,
            Activation('step').nonlinearity
        ], *args, **kwargs)
