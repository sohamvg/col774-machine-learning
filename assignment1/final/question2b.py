import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

m = 10**6
n = 2
theta = np.array([[3], [1], [2]])
x1 = np.random.normal(3, 4, m)
x2 = np.random.normal(-1, 4, m)
x = np.stack((np.ones(m), x1, x2), axis=1).reshape((m, n+1, 1))

noise = np.random.normal(0, 2, m).reshape((m, 1))

np.random.shuffle(x)
y = np.matmul(theta.T, x)

def hypothesis(theta, x):
    return np.matmul(theta.T, x)

def cost(y, x, theta):
    summation = 0
    for i in range(m):
        summation += (y[i] - hypothesis(theta, x[i]))**2
    return summation / (2 * m)

def stochastic_gradient_descent(learning_rate, r, epsilon):
    t = 0
    theta = np.zeros((n+1, 1))
    prev_cost = cost(y, x, theta)
    k = 0
    
    while True:
    
        for b in range(int(m/r)):
            theta_t = theta.copy()
            
            for j in range(n+1):
                summation = 0
                for k in range(r):
                    ik = b*r + k
                    summation += (y[ik] - hypothesis(theta_t, x[ik])) * x[ik][j]
                theta[j] = theta[j] + (learning_rate * summation)
                
            curr_cost = cost(y, x, theta)
            print(b, t, theta, curr_cost)
            
            if abs(curr_cost - prev_cost) < epsilon:
                k += 1
            else:
                k = 0
            
            if k > 5 or t > 1500:
                return theta
            t += 1