import numpy
import pandas
import linweighreg
import matplotlib.pyplot as plot
import math

train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))

model_single = linweighreg.LinearRegression()

print(t_train.shape)

A = numpy.zeros((X_train.T.shape[1], X_train.T.shape[1]), dtype=float)
print(A.shape)

for i in range(A.shape[0]):
    A[i][i] = t_train[i]**2

model_single.fit(X_train, A, t_train)
model_single.predict(X_test)

plot.scatter(model_single.p, t_test)
plot.xlabel('predicted house prices')
plot.ylabel('real house prices')
plot.title('all features')
plot.show()
