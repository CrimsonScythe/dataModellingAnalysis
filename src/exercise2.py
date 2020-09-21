import numpy as np
import matplotlib.pyplot as plot
import math
import scipy.stats

data = np.loadtxt('/home/kamal/MAD/A4/men-olympics-100.txt', delimiter=' ')
X = data[:, 0]
X = X.reshape((len(X), 1))

t = data[:, 1]
t = t.reshape((len(t), 1))


def ComputeMean(cov, X, t, mu):

    cov_z = np.zeros((2, 2))
    cov_z[0][0] = 100
    cov_z[1][1] = 5

    muW = np.dot(cov,
     ((np.dot(X.T, t) / 10) + np.dot(np.linalg.inv(cov_z), mu)))

    return muW

def ComputeCovar(X):
    cov = np.zeros((2, 2))
    cov[0][0] = 100
    cov[1][1] = 5
    # print(cov)
    covW = np.linalg.inv((np.dot(X.T, X) / 10) + np.linalg.inv(cov))
    return covW

def ComputeGauss(X, mean, cov):


    mu0 = np.zeros((2, 1))
    mu0_v = mu0.reshape((2,))

    cov = np.zeros((2, 2))
    cov[0][0] = 100
    cov[1][1] = 5

    # Sample W from a multivariate normal distribution
    va=np.random.multivariate_normal(mu0_v, cov, 10)

    # Compute post.prob desnity using the multivariate normal function with W samples from above
    y=scipy.stats.multivariate_normal.pdf(va, mean=mean, cov=cov)

    return y


mu0 = np.zeros((2, 1))
cov = ComputeCovar(X)
mean = ComputeMean(cov, X, t, mu0)
# Covariane
print("Covariance:\n",cov)
# Mean
print("Mean:\n", mean)
#Posterior Prob. density
mean_v=mean.reshape((2,))
pdf = ComputeGauss(X, mean_v, cov)
print("Post. Prob. Density:\n", pdf.reshape((pdf.shape[0],1)))
