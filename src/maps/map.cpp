//
// Created by mrjak on 14-07-2021.
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../algorithms/SynC.h"
#include "../algorithms/simple_GPU_SynC.cuh"
#include "../algorithms/EGG_SynC.cuh"

////cpp wrapper
//neighborhood ComputeNeighborhood_wrapper(int p, float eps, at::Tensor &D) {
//    return ComputeNeighborhood(p, eps, D.data_ptr<float>(), D.size(0), D.size(1));
//}
//
//void UpdatePoint_wrapper(int x, const neighborhood &N_p, at::Tensor &D_current, at::Tensor &D_next) {
//    UpdatePoint(x, N_p, D_current.data_ptr<float>(), D_next.data_ptr<float>(), D_current.size(0), D_current.size(1));
//}
//
//float ComputeLocationOrder_wrapper(int x, neighborhood N_x, at::Tensor &D) {
//    return ComputeLocationOrder(x, N_x, D.data_ptr<float>(), D.size(0), D.size(1));
//}
//
//clustering synCluster_wrapper(at::Tensor &D) {
//    float eps = 0.1; //todo bad!
//    return synCluster(D.data_ptr<float>(), D.size(0), D.size(1), eps);
//}
//
//outliers Outliers_wrapper(at::Tensor &D, clustering &C) {
//    return Outliers(D.data_ptr<float>(), D.size(0), D.size(1), C);
//}
//
//float NN_wrapper(at::Tensor &D, int k) {
//    return NN(D.data_ptr<float>(), D.size(0), D.size(1), k);
//}
//
//float f_hat_wrapper(int x, at::Tensor &D, at::Tensor &h) {
//    return f_hat(x, D.data_ptr<float>(), D.size(0), D.size(1), h.data_ptr<float>());
//}
//
//float L_wrapper(at::Tensor &D, clustering &M) {
//    return L(D.data_ptr<float>(), D.size(0), D.size(1), M);
//}

std::vector <at::Tensor> DynamicalClustering_wrapper(at::Tensor &D, float eps, float lam) {
    return DynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}

std::vector <at::Tensor> DynamicalClustering_parallel_wrapper(at::Tensor &D, float eps, float lam) {
    return DynamicalClustering_parallel(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}

//clustering SynC_wrapper(at::Tensor &D, int k) {
//    return SynC(D.data_ptr<float>(), D.size(0), D.size(1), k);
//}

std::vector <at::Tensor> FDynamicalClustering_wrapper(at::Tensor &D, float eps, float lam, int B) {
    return FDynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam, B);
}

//cuda wrapper
std::vector <at::Tensor> GPU_DynamicalClustering_wrapper(at::Tensor &D, float eps, float lam) {
    return GPU_DynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}
//
//clustering GPU_DynamicalClustering_GRID1_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID1(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID2_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID2(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID_2D1_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID_2D1(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID_2D2_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID_2D2(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID_STAD0_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID_STAD0(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID_STAD1_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID_STAD1(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering
//GPU_DynamicalClustering_GRID_STAD2_wrapper(at::Tensor &D, float eps, float lam, int cell_factor, int version) {
//    return GPU_DynamicalClustering_GRID_STAD2(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam, cell_factor,
//                                              version);
//}
//
//clustering GPU_DynamicalClustering_list_wrapper(at::Tensor &D, float eps, float lam, int cell_factor, int version) {
//    return GPU_DynamicalClustering_list(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam,
//                                        cell_factor, version);
//}
//
//clustering GPU_DynamicalClustering_GRID_STAD3_wrapper(at::Tensor &D, float eps, float lam) {
//    return GPU_DynamicalClustering_GRID_STAD3(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
//}
//
//clustering GPU_DynamicalClustering_GRID_STAD4_wrapper(at::Tensor &D, float eps, float lam, int cell_factor) {
//    return GPU_DynamicalClustering_GRID_STAD4(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam, cell_factor);
//}
//
//clustering
//GPU_DynamicalClustering_DOUBLE_GRID_wrapper(at::Tensor &D, float eps, float lam, int cell_factor, int version) {
//    return GPU_DynamicalClustering_DOUBLE_GRID(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam, cell_factor,
//                                               version);
//}

std::vector<at::Tensor> EGG_SynC_wrapper(at::Tensor &D, float eps) {
    return EGG_SynC(D.data_ptr<float>(), D.size(0), D.size(1), eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
//m.def("GPU_DynamicalClustering", &GPU_DynamicalClustering, "");
m.def("GPU_DynamicalClustering_wrapper", &GPU_DynamicalClustering_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID1_wrapper", &GPU_DynamicalClustering_GRID1_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID2_wrapper", &GPU_DynamicalClustering_GRID2_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_2D1_wrapper", &GPU_DynamicalClustering_GRID_2D1_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_2D2_wrapper", &GPU_DynamicalClustering_GRID_2D2_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_STAD0_wrapper", &GPU_DynamicalClustering_GRID_STAD0_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_STAD1_wrapper", &GPU_DynamicalClustering_GRID_STAD1_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_STAD2_wrapper", &GPU_DynamicalClustering_GRID_STAD2_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_STAD3_wrapper", &GPU_DynamicalClustering_GRID_STAD3_wrapper, "");
//m.def("GPU_DynamicalClustering_GRID_STAD4_wrapper", &GPU_DynamicalClustering_GRID_STAD4_wrapper, "");
//m.def("GPU_DynamicalClustering_DOUBLE_GRID_wrapper", &GPU_DynamicalClustering_DOUBLE_GRID_wrapper, "");
m.def("EGG_SynC_wrapper", &EGG_SynC_wrapper, "");
//m.def("GPU_DynamicalClustering_list_wrapper", &GPU_DynamicalClustering_list_wrapper, "");
//m.def("SynC", &SynC_wrapper, "");
m.def("DynamicalClustering", &DynamicalClustering, "");
m.def("DynamicalClustering_wrapper", &DynamicalClustering_wrapper, "");
m.def("DynamicalClustering_parallel_wrapper", &DynamicalClustering_parallel_wrapper, "");
m.def("FDynamicalClustering_wrapper", &FDynamicalClustering_wrapper, "");
//m.def("L", &L, "");
//m.def("L_wrapper", &L_wrapper, "");
//m.def("f_hat", &f_hat, "");
//m.def("f_hat_wrapper", &f_hat_wrapper, "");
//m.def("K", &K, "");
//m.def("NN", &NN, "");
//m.def("NN_wrapper", &NN_wrapper, "");
//m.def("Outliers", &Outliers, "");
//m.def("Outliers_wrapper", &Outliers_wrapper, "");
//m.def("synCluster", &synCluster, "");
//m.def("synCluster_wrapper", &synCluster_wrapper, "");
//m.def("ComputeLocationOrder", &ComputeLocationOrder, "");
//m.def("ComputeLocationOrder_wrapper", &ComputeLocationOrder_wrapper, "");
//m.def("UpdatePoint", &UpdatePoint, "");
//m.def("UpdatePoint_wrapper", &UpdatePoint_wrapper, "");
//m.def("ComputeNeighborhood", &ComputeNeighborhood, "");
//m.def("ComputeNeighborhood_wrapper", &ComputeNeighborhood_wrapper, "");
};