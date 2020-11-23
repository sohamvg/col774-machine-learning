import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data_dir = sys.argv[1]
out_dir = sys.argv[2]

with open(os.path.join(out_dir, 'q4f.txt'), "w+") as out_file:
    print("Since we have a quadratic equation in x1 for each x2, we can see 2 quadratic curves out of which one is out quadratic boundary. The quadratic boundary seems to be a little better fit but mostly the linear boundary also resembles the quadratic boundary.", file=out_file)