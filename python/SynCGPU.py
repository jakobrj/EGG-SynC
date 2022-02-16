from torch.utils.cpp_extension import load
import time
import torch
import scipy
import numpy as np

print("Compiling our c++/cuda code... ")
t0 = time.time()
impl = load(name="SynC3",
            sources=[
                "src/maps/map.cpp",
                "src/algorithms/SynC.cpp",
                "src/algorithms/GPU_SynC.cu",
                "src/utils/CPU_math.cpp",
                "src/utils/GPU_utils.cu"
            ], extra_cuda_cflags=["-w"], extra_cflags=["-w"], with_cuda=True)
print("Finished compilation, took: %.4fs" % (time.time() - t0))


def ComputeNeighborhood(p, eps, D):
    return impl.ComputeNeighborhood_wrapper(p, eps, D)


def UpdatePoint(x, N_p, D_current, D_next):
    return impl.UpdatePoint_wrapper(x, N_p, D_current, D_next)


def ComputeLocationOrder(x, N_x, D):
    return impl.ComputeLocationOrder_wrapper(x, N_x, D)


def synCluster(D):
    return impl.synCluster_wrapper(D)


def Outliers(D, Cl):
    return impl.Outliers_wrapper(D, Cl)


def NN(k, D):
    return impl.NN_wrapper(D, k)


def K(x):
    return impl.K(x)


def f_hat(x, D, h):
    return impl.f_hat_wrapper(x, D, h)


def L(D, M):  # todo something seems to be wrong with the cost function???
    return impl.L_wrapper(D, M)


def DynamicalClustering(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_wrapper(D, eps, lam)


def DynamicalClustering_GRID1(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID1_wrapper(D, eps, lam)


def DynamicalClustering_GRID2(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID2_wrapper(D, eps, lam)


def DynamicalClustering_GRID_2D1(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID_2D1_wrapper(D, eps, lam)


def DynamicalClustering_GRID_2D2(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID_2D2_wrapper(D, eps, lam)


def DynamicalClustering_GRID_STAD0(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID_STAD0_wrapper(D, eps, lam)


def DynamicalClustering_GRID_STAD1(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID_STAD1_wrapper(D, eps, lam)


def DynamicalClustering_GRID_STAD2(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=0):
    return impl.GPU_DynamicalClustering_GRID_STAD2_wrapper(D, eps, lam, cell_factor, version)


def DynamicalClustering_GRID_STAD2_1(D, eps, lam=1 - 1e-3, call=None, cell_factor=1):
    return impl.GPU_DynamicalClustering_GRID_STAD2_1_wrapper(D, eps, lam, cell_factor)


def DynamicalClustering_GRID_STAD3(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_GRID_STAD3_wrapper(D, eps, lam)


def DynamicalClustering_GRID_STAD4(D, eps, lam=1 - 1e-3, call=None, cell_factor=1):
    return impl.GPU_DynamicalClustering_GRID_STAD4_wrapper(D, eps, lam, cell_factor)


def DynamicalClustering_list(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=1):
    return impl.GPU_DynamicalClustering_list_wrapper(D, eps, lam, cell_factor, version)


def DynamicalClustering_DOUBLE_GRID(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=0):
    return impl.GPU_DynamicalClustering_DOUBLE_GRID_wrapper(D, eps, lam, cell_factor, version)


def Sync(D, k, call=None):
    return impl.SynC(D, k)
