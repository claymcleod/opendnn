import theano
import numpy as np
from opendnn.utils import data
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
nn.add_layer(Dense(5, name='d2'))
nn.add_layer(Activation('relu'))
nn.add_layer(Dense(2, name='d3'))
nn.add_layer(Activation('softmax'))

nn.compile(loss_fn='categorical_crossentropy', pred_fn='argmax')
nn.train_until_convergence(X, y, threshold=1e-6, verbose=True)
