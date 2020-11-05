# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# load data
x = np.genfromtxt('ass1_data/data/q1/linearX.csv', delimiter=',')
y = np.genfromtxt('ass1_data/data/q1/linearY.csv', delimiter=',')

n = 1
m = np.size(y)

plt.scatter(x, y)
plt.show()

# normalize
x = x - x.mean()
x = x / x.std()


x = x.reshape((m, n))
x = np.hstack((np.ones((m, 1)), x)) # add intercept

y = y.reshape((m, 1))
print(x)
# print(y)

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
    thetas_and_costs = []
    thetas_and_costs.append((theta, prev_cost))
    
    while True:
#         print(t, theta)
        theta_t = theta.copy()
        
        for j in range(n+1):
            summation = 0
            for i in range(m):
                summation += (y[i] - hypothesis(theta_t, x[i])) * x[i][j]
            theta[j] = theta[j] + (learning_rate * summation)
            
        curr_cost = cost(theta)
        thetas_and_costs.append((theta, curr_cost))
        print(t, theta_t, prev_cost, curr_cost)
            
        if abs(curr_cost - prev_cost) < epsilon or t > 100000:
            return theta, thetas_and_costs
        prev_cost = curr_cost
        t += 1

theta, thetas_and_costs = gradient_descent(0.001, 1e-8)
print(thetas_and_costs)

plt.scatter(x.T[1,:], y)
plt.plot(x.T[1,:], [hypothesis(theta, xi) for xi in x], 'C1')
plt.show()

# theta0 = np.linspace(-4, 4, 40)
# theta1 = np.linspace(-4, 4, 40)

# X, Y = np.meshgrid(theta0, theta1)
# # Z = f(X, Y)
# def f(theta0, theta1):
#     theta = np.array([[theta0],
#                         [theta1]])
#     return cost(theta)


# Z = f(X, Y)
# Z = Z.reshape((40, 40))
# fig = plt.figure()

# # ax = plt.axes(projection='3d')
# # ax.contour3D(X, Y, Z, 50, cmap='binary')
# fig, ax = plt.subplots(1, 1)
# ax.contour(X, Y, Z)
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # plt.plot(theta[0], theta[1], markersize=50)
# # plt.show()
# # temp = np.linspace(0,1,20)
# # ax.plot3D(temp, temp, temp, 'red')
# print(theta)

# x0 = []
# x1 = []
# for theta, cost in thetas_and_costs:
#     x0.append(theta[0][0])
#     x1.append(theta[1][0])
# #     fig = plt.figure()

# #     # ax = plt.axes(projection='3d')
# #     # ax.contour3D(X, Y, Z, 50, cmap='binary')
# #     fig, ax = plt.subplots(1, 1)
# #     ax.contour(X, Y, Z)
#     # ax.set_xlabel('x')
#     # ax.set_ylabel('y')
#     # ax.set_zlabel('z')
#     plt.plot(x0, x1, markersize=50)
#     plt.draw()
#     plt.pause(0.2)
    

import time

theta0 = np.linspace(-2, 4, 30)
theta1 = np.linspace(-2, 4, 30)

X, Y = np.meshgrid(theta0, theta1)
# Z = f(X, Y)
def f(theta0, theta1):
    theta = np.array([[theta0],
                        [theta1]])
    return cost(theta)


Z = f(X, Y)
Z = Z.reshape((30, 30))

fig = plt.figure()

ax = plt.axes(projection='3d')

for theta, cost in thetas_and_costs:
#     print(theta, cost)
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    temp = np.linspace(0,1,20)
    ax.scatter3D([theta[0]], [theta[1]], cost, 'red')
    plt.show()
