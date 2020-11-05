#!/usr/bin/env python
# coding: utf-8

# In[94]:


# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# !python --version


# In[95]:


x = np.loadtxt('ass1_data/data/q4/q4x.dat')
y = np.loadtxt('ass1_data/data/q4/q4y.dat', dtype='U')

n = np.size(x, 1)
m = np.size(y)
    
for i in range(m):
    if y[i] == 'Alaska':
        y[i] = 0
    else:
        y[i] = 1

y = y.astype('int32')
        
# normalize
for i in range(n):
    x[:,i] = (x[:,i] - x[:,i].mean()) / x[:,i].std()
    
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
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
    
x = x.reshape((m, n, 1))


# In[140]:


summation1 = np.zeros((n, 1))
summation2 = 0
summation3 = np.zeros((n, 1))
summation4 = 0
summation5 = np.zeros((n, n))

for i in range(m):
    summation1 += int(y[i] == 0) * x[i]
    summation2 += int(y[i] == 0)
    summation3 += int(y[i] == 1) * x[i]
    summation4 += int(y[i] == 1)

phi = summation4 / m
mu0 = summation1 / summation2
mu1 = summation3 / summation4
mu = np.array([mu0, mu1])

for i in range(m):
    summation5 += np.matmul(x[i] - mu[y[i]], (x[i] - mu[y[i]]).T)
    
sigma = summation5 / m
sigma_inv = np.linalg.inv(sigma)
# np.log(phi/(1-phi))

t1 = (mu[0].T @ sigma_inv @ mu[0] - mu[1].T @ sigma_inv @ mu[1])/2
t2 = np.log(phi/(1-phi))
t3 = sigma_inv @ (mu[0] - mu[1])
print(t1, t2, t3, t1 - t2)
# xT = (t1 - t2) @ t3 @ sigma

# t3[0] x1 + t3[1] x2 = t1 - t2

def get_x2(x1):
    return ((t1 - t2) - (t3[0] * x1))/t3[1]

# x[:, 0, 0]
plt.plot(x[:, 0, 0], [get_x2(x1)[0][0] for x1 in x[:, 0, 0]], 'C2')

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
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
    


# In[139]:


summation6 = np.zeros((n, n))
summation7 = np.zeros((n, n))

for i in range(m):
    summation6 += int(y[i] == 0) * (x[i] - mu[y[i]]) @ (x[i] - mu[y[i]]).T
    summation7 += int(y[i] == 1) * (x[i] - mu[y[i]]) @ (x[i] - mu[y[i]]).T
    
sigma0 = summation6 / summation2
sigma1 = summation7 / summation4

print(sigma0, sigma1)

