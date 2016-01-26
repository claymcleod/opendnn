import theano
from opendnn.layers import Activation, Layer

class MReLU(Layer):
    """Essentially a PD controller intuition using the ReLU function"""

    def __init__(self, trainable=False, coefs=[0.5, 0.5], *args, **kwargs):
        assert(len(coefs) == 2), ("Must have 2 coefficients for P, and D.")
        self.coefs = coefs
        self.trainable = trainable
        super(MReLU, self).__init__(*args, **kwargs)

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        self.P = theano.tensor.nnet.relu(X)#0.5 * (X + abs(X))
        self.D = theano.tensor.switch(theano.tensor.le(X, 0), 0, 1)

        if self.trainable:
            self.P_coef = theano.shared(self.coefs[0], name='P_coef{}'.format(layer_num))
            self.D_coef = theano.shared(self.coefs[1], name='D_coef{}'.format(layer_num))
            return self.P_coef * self.P + self.D_coef * self.D
        else:
            return self.coefs[0] * self.P + self.coefs[1] * self.D

    def _get_updates_(self, loss, learning_rate):
        if self.trainable:
            updates = []
            updates.append((self.P_coef, self.P_coef - theano.tensor.grad(loss, self.P_coef) * learning_rate))
            updates.append((self.D_coef, self.D_coef - theano.tensor.grad(loss, self.D_coef) * learning_rate))
            return updates
        else:
            return []
