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
plt.legend(["Alaska {0}", "Canada {1}"])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("q4b")

plt.savefig(os.path.join(out_dir, 'q4b.png'))