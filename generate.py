
from python.data import *
import torch
import sys
import os

n = 1_000_000
d = 4
cl = 4

if len(sys.argv)>1:
    n = int(sys.argv[1])
if len(sys.argv)>2:
    d = int(sys.argv[2])
if len(sys.argv)>3:
    cl = int(sys.argv[3])

filename = "data/n"+str(n)+"d"+str(d)+"cl"+str(cl)+".csv"

#if os.path.exists(filename):
#    exit()

D = torch.from_numpy(min_max_normalize(load_synt_gauss_rnd(d=d, n=n, cl=cl, cl_d=d, std=10., noise=0.))).float()

np.savetxt(filename, D, delimiter=",")