import theano
import theano.tensor
from theano import pp

class NeuralNetwork(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss_fn, pred_fn, learning_rate=0.01):
        self._compile_dims_()
        self._compile_theano_(loss_fn, pred_fn, learning_rate)

    def _compile_dims_(self):
        self.dims = [self.input_dim]
        for layer in self.layers:
            self.dims.append(layer.output_dim)
        self.dims.append(self.output_dim)
        for i in range(2, len(self.dims)):
            if self.dims[i] is None:
                self.dims[i] = self.dims[i - 1]
        self.dims = zip(self.dims, self.dims[1:])

    def _compile_theano_(self, loss_fn, pred_fn, learning_rate):
        X = theano.tensor.vector('X', theano.config.floatX)
        y = theano.tensor.vector('y', theano.config.floatX)
        y_hat = None
        for index, (layer, (input_dim, output_dim)) in enumerate(
                zip(self.layers, self.dims)):
            if y_hat == None:
                y_hat = layer._build_layer_(X, input_dim, output_dim, index)
            else:
                y_hat = layer._build_layer_(y_hat, input_dim, output_dim, index)

        _loss_fn = loss_fn(y_hat, y)
        _pred_fn = pred_fn(y_hat)

        updates = []
        for layer in self.layers:
            updates.extend(layer._get_updates_(_loss_fn, learning_rate))

        self.predict_fn = theano.function([X], _pred_fn)
        self.loss_fn = theano.function([X, y], _loss_fn)
        self.train_fn = theano.function([X, y], updates=updates)
        return

    def train_on_instance(self, X_train, y_train):
        assert hasattr(self, 'train_fn'), (
            "You must first compile the network!")
        self.train_fn(X_train, y_train)

    def get_loss(self, X, y):
        return self.loss_fn(X, y)

    def get_gradients(self, X, y):
        return self.gradient_fn(X, y)
