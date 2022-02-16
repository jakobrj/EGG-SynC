import numpy as np
from python.generator import *


def load_synt_gauss(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
    if cl_d is None:
        cl_d = int(0.8 * d)

    if cl_n is None:
        cl_n = int((1. - noise) * n / cl)

    subspace_clusters = []
    for i in range(cl):
        subspace_clusters.append([cl_n, cl_d, 1, std])

    X, _ = generate_subspacedata_permuted(n, d, subspace_clusters)

    return X


def load_synt_gauss_rnd(d, n, cl, std, cl_n=None, cl_d=None, noise=0.01, re=0):
    if cl_d is None:
        cl_d = int(0.8 * d)

    rnds = []
    total_rnds = 0
    for i in range(cl):
        rnd = random.randint(5, 20)
        total_rnds += rnd
        rnds.append(rnd)

    subspace_clusters = []
    for i in range(cl):
        cl_n_ = cl_n
        if cl_n is None:
            cl_n_ = int((1. - noise) * n * rnds[i] / total_rnds)
        subspace_clusters.append([cl_n_, cl_d, 1, std])

    X, _ = generate_subspacedata_permuted(n, d, subspace_clusters)

    return X


def load_iris():
    X = np.loadtxt("data/iris.data", delimiter=',', skiprows=0)
    return X


def load_pendigits():
    X = np.loadtxt("data/pendigits.tra", delimiter=',', skiprows=0)
    return X


def load_D_spatial_network():
    X = np.loadtxt("data/3D_spatial_network.txt", delimiter=",", skiprows=0, usecols=(1, 2, 3))
    return X


def load_CCPP():
    X = np.loadtxt("data/CCPP.data", delimiter="\t", skiprows=1)
    return X


def load_banknote():
    X = np.loadtxt("data/data_banknote_authentication.txt", delimiter=",", skiprows=0, usecols=(0, 1, 2, 3))
    return X


def load_eb():
    X = np.loadtxt("data/eb.arff", delimiter=",", skiprows=0, usecols=(0,1,4))
    return X


def load_Skin_NonSkin():
    X = np.loadtxt("data/Skin_NonSkin.txt", skiprows=0, usecols=(0, 1, 2))
    return X


def load_Wilt():
    X = np.loadtxt("data/wilt.data", delimiter=",", skiprows=0, usecols=(1, 2, 3, 4, 5))
    return X


def load_Yeast():
    X = np.loadtxt("data/yeast.data", skiprows=0, usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    return X


def min_max_normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)
