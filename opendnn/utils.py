import numpy
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import np_utils


class data(object):

    @staticmethod
    def one_hot_encode(data):
        assert ('int' in str(data.dtype)
                ), ('Only integers can be one hot encoded!')
        return numpy.eye(max(data) + 1)[data]

    @staticmethod
    def get_mnist():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0], 784)
	X_test = X_test.reshape(X_test.shape[0], 784)
	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255
	Y_train = np_utils.to_categorical(y_train, 10).astype('float32')
	Y_test = np_utils.to_categorical(y_test, 10).astype('float32')
	return (X_train, Y_train), (X_test, Y_test)

    @staticmethod
    def get_cifar10():
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	Y_train = np_utils.to_categorical(y_train, 10).astype('float32')
	Y_test = np_utils.to_categorical(y_test, 10).astype('float32')
	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255
	return (X_train, Y_train), (X_test, Y_test)

    @staticmethod
    def get_cifar100():
	(X_train, y_train), (X_test, y_test) = cifar100.load_data()
	Y_train = np_utils.to_categorical(y_train, 100).astype('float32')
	Y_test = np_utils.to_categorical(y_test, 100).astype('float32')
	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255
	return (X_train, Y_train), (X_test, Y_test)
