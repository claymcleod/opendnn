import math
import theano
import theano.tensor
import theano.d3viz as V
import opendnn.loss
import opendnn.initialization
import opendnn.pred

class NeuralNetwork(object):

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss_fn, pred_fn, init_fn='lecun', learning_rate=0.1, use_normal=False):
        self._compile_dims_()
        self._compile_theano_(loss_fn, pred_fn, init_fn, learning_rate, use_normal)

    def _compile_dims_(self):
        self.dims = [self.input_dim]
        for layer in self.layers:
            self.dims.append(layer.output_dim)
        for i in range(2, len(self.dims)):
            if self.dims[i] is None:
                self.dims[i] = self.dims[i - 1]
        self.dims = zip(self.dims, self.dims[1:])

    def _compile_theano_(self, loss_fn, pred_fn, init_fn, learning_rate, use_normal):

        if type(loss_fn) == str:
            loss_fn = opendnn.loss.resolve(loss_fn)

        if type(pred_fn) == str:
            pred_fn = opendnn.pred.resolve(pred_fn)

        if type(init_fn) == str:
            init_fn = opendnn.initialization.resolve(init_fn)


        # Start training
        X = theano.tensor.matrix('X', theano.config.floatX)
        y = theano.tensor.matrix('y', theano.config.floatX)
        y_hat = X

        for index, (layer, (input_dim, output_dim)) in enumerate(
                zip(self.layers, self.dims)):
            y_hat = layer._build_layer_(y_hat, index, input_dim, output_dim, init_fn, use_normal)

        _loss_fn = loss_fn(y_hat, y)
        _pred_fn = pred_fn(y_hat)

        updates = []
        for layer in self.layers:
            _updates = layer._get_updates_(_loss_fn, learning_rate)
            updates.extend(_updates)

        self.raw_output_fn = theano.function([X], y_hat)
        self.predict_fn = theano.function([X], _pred_fn)
        self.loss_fn = theano.function([X, y], _loss_fn)
        self.train_fn = theano.function([X, y], updates=updates)
        return

    def train(self, X, y, batch_size=None):
        assert hasattr(self, 'train_fn'), ("You must first compile the network!")

        if not batch_size:
            self.train_fn(X, y)
        else:
            batch_iters = int(math.ceil(X.shape[0] / batch_size))
            print("Batching")
            for bi in range(batch_iters):
                start = bi*batch_size
                finish = ((bi+1)*batch_size)
                self.train(X[start:finish], y[start:finish])

    def train_until_convergence(self, X, y, max_iterations=None, step=100, threshold=1e-3,
                               verbose=False, track_loss=False):
        iters = 0
        last_loss = 0
        losses = []

        while max_iterations is None or iters < max_iterations:
            for x in range(step+1):
                self.train(X, y)

            this_loss = self.get_loss(X, y)
            if track_loss:
                losses.append(this_loss)

            if verbose:
                print("Iteration {} - Loss: {}".format(iters, this_loss))

            if (abs(last_loss - this_loss) <= threshold):
                break

            last_loss = this_loss
            iters = iters + step

        if verbose:
            print("Converged at iteration {}".format(iters))

        return iters, losses

    def get_loss(self, X, y):
        return self.loss_fn(X, y)

    def print_model(self):
        for layer in self.layers:
            print('- {}'.format(layer))
