import numpy
import theano
import theano.tensor

class Layer(object):

    def __init__(self, output_dim=None, name=''):
        self.output_dim = output_dim
        self.name = name

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        raise NotImplementedError(
            "You must implement _build_layer_()!")

    def _get_updates_(self, loss, learning_rate):
        raise NotImplementedError(
            "You must implement _get_updates_()!")


class Dense(Layer):

    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_fn = init_fn
        self.W = theano.shared(init_fn(input_dim, output_dim, use_normal),
                               name='Layer{}_W'.format(layer_num))
        self.B = theano.shared(numpy.zeros(output_dim),
                               name='Layer{}_B'.format(layer_num))
        return X.dot(self.W) + self.B

    def _get_updates_(self, loss, learning_rate):
        updates = []
        updates.append(
            (self.W,
             self.W -
             theano.tensor.grad(
                 loss,
                 self.W) *
                learning_rate))
        updates.append(
            (self.B,
             self.B -
             theano.tensor.grad(
                 loss,
                 self.B) *
                learning_rate))
        return updates

    def __repr__(self):
        return 'Dense {} x {} ({})'.format(self.input_dim, self.output_dim, self.init_fn)


class Activation(Layer):

    def __init__(self, activation, *args, **kwargs):
        activation = activation.lower()
        if activation == 'relu':
            assert hasattr(theano.tensor.nnet, 'relu'), (
                '\'relu\' not available in this version of theano.'
                ' Please upgrade by using the following command:\n'
                'pip install git+git://github.com/Theano/Theano.git --upgrade')
            self.nonlinearity = theano.tensor.nnet.relu
        elif activation == 'softmax':
            self.nonlinearity = theano.tensor.nnet.softmax
        else:
            raise RuntimeError(
                "Unsupported activation type: {}".format(activation))
        super(Activation, self).__init__(*args, **kwargs)

    def _build_layer_(self, X, layer_num, input_dim, output_dim, init_fn, use_normal):
        self.input_dim = input_dim
        self.output_dim = output_dim
        return self.nonlinearity(X)

    def _get_updates_(self, loss, learning_rate):
        return []

    def __repr__(self):
        return 'Activation {} x {}'.format(self.input_dim, self.output_dim)
