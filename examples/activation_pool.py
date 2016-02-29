from __future__ import print_function

import sys
from time import time
from opendnn.utils import data
from opendnn.layers import Dense, Activation
from opendnn.experiments import MReLU
from opendnn.models import NeuralNetwork


def get_layer_for_setting(nn, setting):
    if setting == 'relu':
        nn.add_layer(Activation('relu'))
    elif setting == 'pap':
        nn.add_layer(MReLU(trainable=True))
    elif setting == 'aphalf':
        nn.add_layer(MReLU(trainable=False, coefs=[0.5, 0.5]))
    elif setting == 'apstaggered':
        nn.add_layer(MReLU(trainable=False, coefs=[0.8, 0.2]))
    else:
        print("Invalid setting: {}".format(setting))
        sys.exit(1)

if len(sys.argv) <= 3:
    print("Must include command line arguments!")
    sys.exit(1)

setting = sys.argv[1].lower()
filepath = sys.argv[2]
learning_rate = float(sys.argv[3])

print('Setting: {}'.format(setting))
print('Learning Rate: {}'.format(learning_rate))
print('Filepath: {}'.format(filepath))
print()

f = open(filepath, 'wa')

num_layers = 5
num_hidden_nodes = 784
num_training_iterations = 1000

# Get the cifar10 dataset
print("Getting data...")
(X_train, y_train), (X_test, y_test) = data.get_mnist()

# Compile network
print("Compiling model...", end='')
nn = NeuralNetwork(X_train.shape[1])
for x in range(num_layers):
    nn.add_layer(Dense(num_hidden_nodes))
    get_layer_for_setting(nn, setting)
nn.add_layer(Dense(10))
nn.add_layer(Activation('softmax'))
nn.compile(loss_fn='categorical_crossentropy', init_fn='lecun', pred_fn='argmax',
           learning_rate=learning_rate, use_normal=True)
print('finished!')

ap_loss = []
for i in range(1, num_training_iterations+1):
    start = time()
    nn.train(X_train, y_train)
    time_elapsed = time() - start
    s = "{},{},{}".format(i, nn.get_accuracy(X_train, y_train), time_elapsed)
    print(s)
    f.write(s+'\n')
    #ap_loss.append()
    #print(ap_loss[-1])

# Plotting
#import matplotlib.pyplot as plt
#ap_, = plt.plot(ap_loss, 'g.', label="activationpool")
#relu_, = plt.plot(relu_loss, 'b^', label="relu")
#plt.title("Activation Pool vs. ReLU loss over time")
#plt.xlabel("Iterations")
#plt.ylabel("Loss fn (categorical crossentropy)")
#plt.legend(handles=[ap_, relu_])
#plt.show()
