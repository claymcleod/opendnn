import theano
import numpy as np
from opendnn import loss
from opendnn import pred
from opendnn.utils import data
from opendnn.initialization import glorot
from opendnn.layers import Dense, Activation
from opendnn.models import NeuralNetwork

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])
y = data.one_hot_encode(y)

nn = NeuralNetwork(2)
nn.add_layer(Dense(5, name='d1'))
nn.add_layer(Activation('relu'))
nn.add_layer(Dense(2, name='d2'))
nn.add_layer(Activation('softmax'))

nn.compile(loss_fn=loss.categorical_crossentropy, pred_fn=pred.argmax, init_fn=glorot)
for x in xrange(10000):
    nn.train(X, y)

    if x % 100 == 0:
        print('Iteration {}: Prediction {} - Loss {}'.format(x, nn.predict_fn(X), nn.loss_fn(X, y)))
