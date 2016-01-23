import theano
from opendnn import loss
from opendnn import pred
from opendnn.layers import Dense, Activation
from opendnn.models import NeuralNetwork

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = [[0],
     [1],
     [1],
     [0]]

nn = NeuralNetwork(2, 1)
nn.add_layer(Dense(40, name='d1'))
nn.add_layer(Dense(1, name='d2'))
nn.add_layer(Activation('softmax'))

nn.compile(loss_fn=loss.categorical_crossentropy, pred_fn=pred.argmax)
for x in xrange(10000):
    for i in xrange(len(y)):
        nn.train_on_instance(X[i], y[i])

    if x % 100 == 0:
        for layer in nn.layers:
            if type(layer) == Dense:
                print(layer.W.get_value())
                #theano.printing.Print("Layer W:")(layer.W)
        answers = []
        for i in xrange(len(X)):
            answers.append(nn.predict_fn(X[i])[0])
        print(answers)
