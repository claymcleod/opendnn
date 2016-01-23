import numpy


class data(object):

    @staticmethod
    def one_hot_encode(data):
        assert ('int' in str(data.dtype)
                ), ('Only integers can be one hot encoded!')
        return numpy.eye(max(data) + 1)[data]
