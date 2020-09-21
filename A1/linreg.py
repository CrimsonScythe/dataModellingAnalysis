import numpy
import matplotlib.pyplot as plot

class LinearRegression():


    def __init__(self):

        pass

    def fit(self, X, t):

        # reshape array in case X only has one attribute
        if X.shape == (X.shape[0],) or X.shape == ():
            X = X.reshape((X.shape[0],1))

        X = numpy.concatenate((numpy.ones((X.shape[0],1)), X), axis=1)

        result = numpy.dot(X.T, t)
        arg = numpy.dot(X.T, X)

        self.w = numpy.linalg.solve(arg,result)
        self.t = t


    def predict(self, X):

        trans =numpy.concatenate((numpy.ones((X.shape[0],1)), X), axis=1)
        predictions = numpy.dot(trans, self.w)
        self.p = predictions
