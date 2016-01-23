from opendnn.models import NeuralNetwork
from opendnn.layers import Dense, Activation
from opendnn.loss import categorical_crossentropy
from opendnn.pred import argmax

def test_construct_nn():
    nn = NeuralNetwork(2)
    nn.add_layer(Dense(40))
    nn.add_layer(Activation('relu'))
    nn.add_layer(Dense(2))
    nn.add_layer(Activation('softmax'))

    assert len(nn.layers) == 4
    assert not hasattr(nn, 'train_fn')

    nn.compile(categorical_crossentropy, argmax)

    assert hasattr(nn, 'train_fn')
