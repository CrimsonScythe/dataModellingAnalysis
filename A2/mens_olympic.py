import numpy
import pandas
import reg_linreg
import matplotlib.pyplot as plot
import math

train_data = numpy.loadtxt("/home/kamal/MAD/A2/men-olympics-100.txt", delimiter=" ")

# extract first column
X = train_data[:, 0]
# reshape to get n dimensional array
X = X.reshape((len(X), 1))

# append ones to get w0
X = numpy.concatenate((numpy.ones((X.shape[0],1)), X), axis=1)

# extract target
target = train_data[:, 1]
t=target

model_single = reg_linreg.RegularizedLinearRegression()

lam_arr = numpy.logspace(-8, 0, 100, base=10)
err_arr = numpy.arange(lam_arr.shape[0], dtype = numpy.float)

for i in range(lam_arr.shape[0]):
    err_arr[i] = model_single.LOOCV(X, target, lam_arr[i])

bestLambda = lam_arr[numpy.argmin(err_arr, axis=0)]
print("Best lambda value first order: %.10f" % bestLambda)


wb = numpy.linalg.solve(
numpy.dot(X.T, X) + numpy.dot(numpy.dot(X.shape[0], bestLambda), numpy.identity(X.shape[1]))
,
numpy.dot(X.T, t))

print("w corresponding to best value of lambda:" , wb)


w0 = numpy.linalg.solve(
numpy.dot(X.T, X) + numpy.dot(numpy.dot(X.shape[0], 0), numpy.identity(X.shape[1]))
,
numpy.dot(X.T, t))

print("w corresponding to lambda 0:" , w0)
plot.xscale('log')
plot.scatter(lam_arr, err_arr)
plot.show()

f_order = train_data[:, 0]
f_order = f_order.reshape((len(f_order), 1))

other_orders = numpy.zeros((f_order.shape[0],3), dtype=float)
for i in range(other_orders.shape[1]):
    for j in range(other_orders.shape[0]):
        other_orders[j][i] = f_order[j]**(i+2)


X = numpy.concatenate((X, other_orders), axis=1)


l = numpy.logspace(-8, 0, 100, base=10)
err = numpy.arange(l.shape[0], dtype = numpy.float)

for j in range(l.shape[0]):
    err[j] = model_single.LOOCV(X, target, l[j])

bestLambdaf = l[numpy.argmin(err, axis=0)]

print("Best lambda value fourth order: %.10f" % bestLambdaf)


wb = numpy.linalg.solve(
numpy.dot(X.T, X) + numpy.dot(numpy.dot(X.shape[0], bestLambdaf), numpy.identity(X.shape[1]))
,
numpy.dot(X.T, t))

print("w corresponding to best value of lambda:" , wb)

w0 = numpy.linalg.solve(
numpy.dot(X.T, X) + numpy.dot(numpy.dot(X.shape[0], 0), numpy.identity(X.shape[1]))
,
numpy.dot(X.T, t))

print("w corresponding to lambda 0:\n" , w0)
plot.xscale('log')
plot.scatter(l, err)
plot.show()
