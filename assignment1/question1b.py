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
x_orig = np.genfromtxt(os.path.join(data_dir, 'linearX.csv'), delimiter=',')
y_orig = np.genfromtxt(os.path.join(data_dir, 'linearY.csv'), delimiter=',')
x = np.genfromtxt(os.path.join(data_dir, 'linearX.csv'), delimiter=',')
y = np.genfromtxt(os.path.join(data_dir, 'linearY.csv'), delimiter=',')

n = 1
m = np.size(y)

x = (x - x.mean()) / x.std() # normalize
x = x.reshape((m, n))
x = np.hstack((np.ones((m, 1)), x)) # add intercept

x = x.reshape((m, n+1, 1))
y = y.reshape((m, 1))

def hypothesis(theta, x):
    return np.matmul(theta.T, x)

def cost(theta):
    summation = 0
    for i in range(m):
        summation += (y[i] - hypothesis(theta, x[i]))**2
    return summation / (2 * m)

def gradient_descent(learning_rate, epsilon):
    t = 0
    theta = np.zeros((n+1, 1))
    prev_cost = cost(theta)
    all_thetas = []
    all_costs = []
    
    while True:
        theta_t = theta.copy()
        all_thetas.append(theta_t)
        all_costs.append(prev_cost)

        for j in range(n+1):
            summation = 0
            for i in range(m):
                summation += (y[i] - hypothesis(theta_t, x[i])) * x[i][j]
            theta[j] = theta[j] + (learning_rate * summation)
            
        curr_cost = cost(theta)
            
        if abs(curr_cost - prev_cost) < epsilon or t > 100000:
            return theta, np.array(all_thetas), np.array(all_costs)
        prev_cost = curr_cost
        t += 1


theta, all_thetas, all_costs = gradient_descent(0.001, 1e-8)

# plotting
plt.scatter(x_orig, y_orig)
plt.plot(x_orig, [hypothesis(theta, xi)[0] for xi in x], 'C1')

plt.xlabel("x")
plt.ylabel("y")
plt.title("q1b")
plt.savefig(os.path.join(out_dir, 'q1b.png'))