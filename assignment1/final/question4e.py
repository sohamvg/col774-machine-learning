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
x = np.loadtxt(os.path.join(data_dir, 'q4x.dat'))
y = np.loadtxt(os.path.join(data_dir, 'q4y.dat'), dtype='U')

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

x = x.reshape((m, n, 1))

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

summation6 = np.zeros((n, n))
summation7 = np.zeros((n, n))

for i in range(m):
    summation6 += int(y[i] == 0) * (x[i] - mu[y[i]]) @ (x[i] - mu[y[i]]).T
    summation7 += int(y[i] == 1) * (x[i] - mu[y[i]]) @ (x[i] - mu[y[i]]).T
    
sigma0 = summation6 / summation2
sigma1 = summation7 / summation4


sigma0_inv = np.linalg.inv(sigma0)
sigma1_inv = np.linalg.inv(sigma1)

LHS = np.log(phi/(1-phi)) + (1/2)*np.log(np.linalg.det(sigma0)/np.linalg.det(sigma1))

c = (mu1.T @ sigma1_inv @ mu1) - (mu0.T @ sigma0_inv @ mu0)

x2 = 1

M = sigma1_inv - sigma0_inv
a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]

e = sigma1_inv @ mu1
f = sigma0_inv @ mu0

coeff2 = a / 2
coeff1 = (b + c) * x2 - e[0] + f[0]
coeff0 = (d * (x2**2) / 2) - (e[1] * x2) + (f[1] * x2) + c - LHS

def get_x1(x2):
    coeff_2 = a / 2
    coeff_1 = ((b + c) * x2)/2 - e[0] + f[0]
    coeff_0 = (d * (x2**2) / 2) - (e[1] * x2) + (f[1] * x2) + c - LHS
    return np.roots([coeff_2, coeff_1, coeff_0])

def get_x2(x1):
    coeff_2 = d / 2
    coeff_1 = (((b + c) * x1) / 2) - e[1] + f[1]
    coeff_0 = (a * (x1**2) / 2) - (e[0] * x1) + (f[0] * x1) + c - LHS
    return np.roots([coeff_2, coeff_1, coeff_0])

# plt.plot(x[:, 0, 0], [get_x2(x1)[1] for x1 in x[:, 0, 0]], 'C2')
plt.scatter([get_x1(x2)[1] for x2 in x[:, 1, 0]], x[:, 1, 0])

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

plt.savefig(os.path.join(out_dir, 'q4e.png'))
