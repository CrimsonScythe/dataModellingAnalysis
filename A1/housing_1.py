import numpy
import matplotlib.pyplot as plot
import math
# load data
train_data = numpy.loadtxt("/home/kamal/MAD/A1/boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("/home/kamal/MAD/A1/boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean = numpy.mean(t_train)
print("Mean price is: %f" % mean)

mean_vector = numpy.full((t_train.shape[0], 1), mean)

# (b) RMSE function
def rmse(t, tp):
    RMSE=0
    for i in range(t_train.shape[0]):
        RMSE+=(abs(t[i] - tp[i])**2)
    RMSE=RMSE/t_train.shape[0]
    return math.sqrt(RMSE)

print("RMSE is: %f" % rmse(mean_vector, t_test))
# (c) visualization of results
plot.xlabel('real house prices')
plot.ylabel('predicted house prices')
plot.title('model using only the mean')
plot.scatter(t_test, mean_vector)
plot.show()
