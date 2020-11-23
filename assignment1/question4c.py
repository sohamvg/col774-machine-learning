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

for i in range(m):
    summation5 += np.matmul(x[i] - mu[y[i]], (x[i] - mu[y[i]]).T)
    
sigma = summation5 / m
sigma_inv = np.linalg.inv(sigma)

t1 = (mu[0].T @ sigma_inv @ mu[0] - mu[1].T @ sigma_inv @ mu[1])/2
t2 = np.log(phi/(1-phi))
t3 = sigma_inv @ (mu[0] - mu[1])

def get_x2(x1):
    "t3[0] x1 + t3[1] x2 = t1 - t2"
    return ((t1 - t2) - (t3[0] * x1))/t3[1]

plt.plot(x[:, 0, 0], [get_x2(x1)[0][0] for x1 in x[:, 0, 0]], color="red", label="boundary")

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
plt.scatter(x1_0, x2_0, marker='o', label="Alaska")
plt.scatter(x1_1, x2_1, marker='^', label="Canada")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("q4c")
plt.legend()

plt.savefig(os.path.join(out_dir, 'q4c.png'))