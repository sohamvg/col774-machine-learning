import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

m = 10**6
n = 2
theta = np.array([[3], [1], [2]])

# sampling normal distribution
x1 = np.random.normal(3, 4, m)
x2 = np.random.normal(-1, 4, m)
x = np.stack((np.ones(m), x1, x2), axis=1).reshape((m, n+1, 1))

noise = np.random.normal(0, 2, m).reshape((m, 1))