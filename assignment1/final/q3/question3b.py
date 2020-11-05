import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data_dir = sys.argv[1]
out_dir = sys.argv[2]

# load data
x = np.genfromtxt(os.path.join(data_dir, 'logisticX.csv'), delimiter=',')
y = np.genfromtxt(os.path.join(data_dir, 'logisticY.csv'), delimiter=',')

n = x[0,:].size
m = y.size

# normalize
for i in range(n):
    x[:,i] = (x[:,i] - x[:,i].mean()) / x[:,i].std()

x = np.hstack((np.ones((m, 1)), x)) # add intercept terms
x = x.reshape((m, n+1, 1))
y = y.reshape((m, 1))


def g(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(theta, x):
    return g(np.matmul(theta.T, x))

def LL(theta):
    "Log likelihood"
    summation = 0
    for i in range(m):
        summation += y[i]*np.log(hypothesis(theta, x[i])) + (1 - y[i])*np.log(1 - hypothesis(theta, x[i]))
    return summation

def derivative_LL(theta):
    "Derivative of Log likelihood wrt. theta"
    delta = np.zeros((n+1, 1))
    for j in range(n+1):
        summation = 0
        for i in range(m):
            summation += (y[i] - hypothesis(theta, x[i])) * x[i][j]
        delta[j] = summation
    return delta


def hessian(theta):
    H = np.zeros((n+1, n+1))
    for j in range(n+1):
        for k in range(j, n+1):
            summation = 0
            for i in range(m):
                summation += hypothesis(theta, x[i])*(1 - hypothesis(theta, x[i])) * x[i][j] * x[i][k]
            H[j][k] = summation
            H[k][j] = summation
    return H

def newtons_method(epsilon):
    t = 0
    theta = np.zeros((n+1, 1))
    prev_LL = LL(theta)
    
    while True:
        theta = theta + np.matmul((np.linalg.inv(hessian(theta))), derivative_LL(theta))
        curr_LL = LL(theta)
        
        if abs(prev_LL - curr_LL) < epsilon or t > 50:
            return theta
        prev_LL = curr_LL
        t += 1

theta = newtons_method(1e-8)

###################################################

# load original data
x = np.genfromtxt(os.path.join(data_dir, 'logisticX.csv'), delimiter=',')
y = np.genfromtxt(os.path.join(data_dir, 'logisticY.csv'), delimiter=',')

n = x[0,:].size
m = y.size

def get_x2(theta, x1):
    "theta0 + theta1*x1 + theta2*x2 = 0"
    return -(theta[0] + theta[1]*x1)/theta[2]
    
x2_pred = [get_x2(theta, xi) for xi in x[:, 0]]
    
x1_0 = []
x2_0 = []
x1_1 = []
x2_1 = []
for i in range(m):
    if y[i] == 0:
        x1_0.append(x[i][0])
        x2_0.append(x[i][1])
    else:
        x1_1.append(x[i][0])
        x2_1.append(x[i][1])

plt.scatter(x1_0, x2_0, marker='o')
plt.scatter(x1_1, x2_1, marker='^')
plt.legend(["0", "1"])
plt.plot(x[:,0], x2_pred, 'C2')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("q3b")

plt.savefig(os.path.join(out_dir, 'q3b.png'))