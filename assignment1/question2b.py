import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data_dir = sys.argv[1]
out_dir = sys.argv[2]

m = 10**6
n = 2
theta = np.array([[3], [1], [2]])

# sampling normal distribution
x1 = np.random.normal(3, 4, m)
x2 = np.random.normal(-1, 4, m)
x = np.stack((np.ones(m), x1, x2), axis=1).reshape((m, n+1, 1))

noise = np.random.normal(0, 2, m).reshape((m, 1))

# shuffle training data
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
    kk = 0
    epoch = 0
    
    while True:
    
        for b in range(int(m/r)):
            theta_t = theta.copy()
            
            for j in range(n+1):
                summation = 0
                for k in range(r):
                    ik = b*r + k
                    summation += (y[ik] - hypothesis(theta_t, x[ik])) * x[ik][j]
                theta[j] = theta[j] + (learning_rate * summation)
                
            if ((abs(theta - theta_t) <= epsilon).all()):
                kk += 1
            else:
                kk = 0
            
            if kk > 2 or epoch > 8:
                return theta
            t += 1
        epoch += 1

r_list = [1, 100, 10000, 1000000]
learning_rate_list = [0.001, 0.0005, 0.000001, 0.000001]
epsilon_list = [1e-5, 1e-5, 1e-4, 1e-2]

for i in range(4):
    r = r_list[i]
    learning_rate = learning_rate_list[i]
    epsilon = epsilon_list[i]

    theta = stochastic_gradient_descent(learning_rate, r, epsilon)

    with open(os.path.join(out_dir, 'q2b_' + str(r) + '.txt'), "w+") as out_file:
        print("\nr = " + str(r) + "\n", file=out_file)
        print("theta = ", theta, file=out_file)