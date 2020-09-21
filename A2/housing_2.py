import numpy
import pandas
import linreg
import matplotlib.pyplot as plot
import math

def rmse(t, tp):
    sum=0
    for i in range(t_train.shape[0]):
        sum+=(abs(t[i] - tp[i])**2)
    sum=sum/t_train.shape[0]
    return math.sqrt(sum)
# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

model_single = linreg.LinearRegression()
# (b) fit linear regression using only the first feature
model_single.fit(X_train[:,0], t_train)
model_single.predict(X_test[:, [1]])
plot.subplot(1, 2, 1)
plot.scatter(model_single.p, t_test)
plot.xlabel('predicted house prices')
plot.ylabel('real house prices')
plot.title('single feature')
print("RMSE single feature: %f" % rmse(t_test, model_single.p))
print("single feature weights:")
print(model_single.w)
# (c) fit linear regression model using all features
model_single.fit(X_train, t_train)
model_single.predict(X_test)
# (d) evaluation of results
plot.subplot(1, 2, 2)
plot.scatter(model_single.p, t_test)
plot.xlabel('predicted house prices')
plot.ylabel('real house prices')
plot.title('all features')
print("RMSE all features: %f" % rmse(t_test, model_single.p))
print("all feature weights:")
print(model_single.w)
plot.show()
