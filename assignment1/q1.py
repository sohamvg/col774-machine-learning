#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# !python --version


# In[3]:


m = 100
n = 1
learning_rate = 0.0001


# In[4]:


x_data = np.genfromtxt('ass1_data/data/q1/linearX.csv', delimiter=',')
y_data = np.genfromtxt('ass1_data/data/q1/linearY.csv', delimiter=',')

# normalize
x_data = x_data - x_data.mean()
x_data = x_data / x_data.std()

plt.scatter(x_data, y_data)
plt.show()

x_data = x_data.reshape((m, n))
x_data = np.hstack((np.ones((m, 1)), x_data)) # add intercept

y_data = y_data.reshape((m, 1))
# print(x_data)
# print(y_data)


# In[5]:


def hypothesis(theta, x):
    return np.matmul(theta.T, x)


# In[6]:


theta = np.zeros((n+1, 1))
# print(theta)

hypothesis(theta, x_data[2])

# [hypothesis(theta, xi) for xi in x_data]


# In[7]:


def cost(theta):
    summation = 0
    for i in range(m):
        summation += (y_data[i] - hypothesis(theta, x_data[i]))**2
    return summation / (2 * m)
        
cost(theta)


# In[8]:


def gradient_descent():
    t = 0
    theta = np.zeros((n+1, 1))
    prev_cost = cost(theta)
    
    while True:
#         print(t, theta)
        theta_t = theta.copy()
        
        for j in range(n+1):
            summation = 0
            for i in range(m):
                summation += (y_data[i] - hypothesis(theta_t, x_data[i])) * x_data[i][j]
            theta[j] = theta[j] + (learning_rate * summation)
            
        curr_cost = cost(theta)
        print(t, theta_t, prev_cost, curr_cost)
            
        if abs(curr_cost - prev_cost) < 1e-8 or t > 100000:
            return theta
        prev_cost = curr_cost
        t += 1


# In[9]:


theta = gradient_descent()
print(theta)


# In[10]:


plt.scatter(x_data.T[1,:], y_data)
plt.plot(x_data.T[1,:], [hypothesis(theta, xi) for xi in x_data], 'C1')
plt.show()


# In[35]:


theta0 = np.linspace(-10, 10, 30)
theta1 = np.linspace(-10, 10, 30)

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
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

