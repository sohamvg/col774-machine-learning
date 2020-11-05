#!/usr/bin/env python
# coding: utf-8

# In[125]:


# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# !python --version


# In[126]:


m = 10**6
n = 2
theta = np.array([[3], [1], [2]])
x1 = np.random.normal(3, 4, m)
x2 = np.random.normal(-1, 4, m)
x = np.stack((np.ones(m), x1, x2), axis=1).reshape((m, n+1, 1))

noise = np.random.normal(0, 2, m).reshape((m, 1))

# s = x1
# mu = 3
# sigma = 4
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()


# In[127]:


np.random.shuffle(x)
y = np.matmul(theta.T, x)
plt.scatter(x[:,1,:], y)
plt.show()


# In[128]:


def hypothesis(theta, x):
    return np.matmul(theta.T, x)


# In[129]:


def cost(y, x, theta):
    summation = 0
    for i in range(m):
        summation += (y[i] - hypothesis(theta, x[i]))**2
    return summation / (2 * m)


# In[133]:


def stochastic_gradient_descent(learning_rate, r, epsilon):
    t = 0
    theta = np.zeros((n+1, 1))
    prev_cost = cost(y, x, theta)
    k = 0
#     thetas_and_costs = []
#     thetas_and_costs.append((theta, prev_cost))
    
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
#             prev_cost = curr_cost
            t += 1


# In[134]:


theta = stochastic_gradient_descent(0.0001, 100, 1e-5)

