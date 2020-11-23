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

with open(os.path.join(out_dir, 'q4a.txt'), "w+") as out_file:
    print("mu0", mu0, "\n", file=out_file)
    print("mu1", mu1, "\n", file=out_file)
    print("sigma", sigma, "\n", file=out_file)