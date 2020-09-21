import numpy
import matplotlib.pyplot as plot

class RegularizedLinearRegression():

    def __init__(self):

        pass

    def LOOCV(self, one_feature, target, lam):

        sum=numpy.float32(0.0)


        for i in range(one_feature.shape[0]):

            X = one_feature
            t = target
            X_del = numpy.delete(X, i, 0)
            t_del = numpy.delete(t, i, 0)


            w = numpy.linalg.solve(
                numpy.dot(X_del.T, X_del) + numpy.dot(numpy.dot(X_del.shape[0], lam), numpy.identity(one_feature.shape[1]))
                ,
                numpy.dot(X_del.T, t_del))

            sum += (target[i] - numpy.dot(w.T, X[i, :]))**2
        error = sum / one_feature.shape[0]
        return error
