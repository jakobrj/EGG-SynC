from torch.utils.cpp_extension import load
import time

print("Compiling our c++/cuda code... ")
t0 = time.time()
impl = load(name="SynC8",
            sources=[
                "src/maps/map.cpp",
                "src/algorithms/SynC.cpp",
                "src/algorithms/EGG_SynC.cu",
                "src/algorithms/simple_GPU_SynC.cu",
                "src/utils/CPU_math.cpp",
                "src/utils/GPU_utils.cu"
            ],
            extra_cuda_cflags=["-w", "--std=c++14", "--ptxas-options=-v"],
            extra_cflags=["-w", "--std=c++17", "-fopenmp"],
            with_cuda=True)
print("Finished compilation, took: %.4fs" % (time.time() - t0))


def DynamicalClustering(D, eps, lam=1 - 1e-3, call=None):
    return impl.DynamicalClustering_wrapper(D, eps, lam)


def DynamicalClustering_parallel(D, eps, lam=1 - 1e-3, call=None):
    return impl.DynamicalClustering_parallel_wrapper(D, eps, lam)


def FDynamicalClustering(D, eps, lam=1 - 1e-3, call=None, B=50):
    return impl.FDynamicalClustering_wrapper(D, eps, lam, B)


def GPU_DynamicalClustering(D, eps, lam=1 - 1e-3, call=None):
    return impl.GPU_DynamicalClustering_wrapper(D, eps, lam)


#
# def GPU_DynamicalClustering_GRID1(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID1_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID2(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID2_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_2D1(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID_2D1_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_2D2(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID_2D2_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_STAD0(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID_STAD0_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_STAD1(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID_STAD1_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_STAD2(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=0):
#     return impl.GPU_DynamicalClustering_GRID_STAD2_wrapper(D, eps, lam, cell_factor, version)
#
#
# def GPU_DynamicalClustering_GRID_STAD2_1(D, eps, lam=1 - 1e-3, call=None, cell_factor=1):
#     return impl.GPU_DynamicalClustering_GRID_STAD2_1_wrapper(D, eps, lam, cell_factor)
#
#
# def GPU_DynamicalClustering_GRID_STAD3(D, eps, lam=1 - 1e-3, call=None):
#     return impl.GPU_DynamicalClustering_GRID_STAD3_wrapper(D, eps, lam)
#
#
# def GPU_DynamicalClustering_GRID_STAD4(D, eps, lam=1 - 1e-3, call=None, cell_factor=1):
#     return impl.GPU_DynamicalClustering_GRID_STAD4_wrapper(D, eps, lam, cell_factor)
#
#
# def GPU_DynamicalClustering_list(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=1):
#     return impl.GPU_DynamicalClustering_list_wrapper(D, eps, lam, cell_factor, version)
#
#
def GPU_DynamicalClustering_DOUBLE_GRID(D, eps, lam=1 - 1e-3, call=None, cell_factor=1, version=0):
    return impl.GPU_DynamicalClustering_DOUBLE_GRID_wrapper(D, eps, lam, cell_factor, version)


"""
"""


def SynC(D, eps):
    return DynamicalClustering(D, eps, lam=1 - 1e-3)


def SynC_parallel(D, eps):
    return DynamicalClustering_parallel(D, eps, lam=1 - 1e-3)


def FSynC(D, eps, B=100):
    return FDynamicalClustering(D, eps, lam=1 - 1e-3, B=B)


def GPU_SynC(D, eps, version=5):
    return GPU_DynamicalClustering_DOUBLE_GRID(D, eps, version=version)


def simple_GPU_SynC(D, eps):
    return GPU_DynamicalClustering(D, eps)


def EGG_SynC(D, eps):
    return impl.EGG_SynC_wrapper(D, eps)
