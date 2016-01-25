import theano
from opendnn.layers import Activation, Layer

class Momentum(Layer):

    def __init__(self, use_normal=False, coefs=[0.3, 0.4, 0.3], *args, **kwargs):
        assert(len(coefs) == 3), ("Must have 3 coefficients for I, P, and D.")
        self.coefs = coefs
        super(Momentum, self).__init__(*args, **kwargs)

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        self.I = theano.tensor.switch(theano.tensor.le(X, 0), 0, X ** 2 / 2)
        self.P = 0.5 * (X + abs(X))
        self.D = theano.tensor.switch(theano.tensor.le(X, 0), 0, 1)

        return self.coefs[0] * self.I + self.coefs[1] * self.P + self.coefs[2] * self.D

    def _get_updates_(self, loss, learning_rate):
        return []
