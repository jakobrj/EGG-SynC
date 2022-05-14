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


std::vector <at::Tensor> DynamicalClustering_wrapper(at::Tensor &D, float eps, float lam) {
    return DynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}

std::vector <at::Tensor> DynamicalClustering_parallel_wrapper(at::Tensor &D, float eps, float lam) {
    return DynamicalClustering_parallel(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}
std::vector <at::Tensor> FDynamicalClustering_wrapper(at::Tensor &D, float eps, float lam, int B) {
    return FDynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam, B);
}

std::vector <at::Tensor> GPU_DynamicalClustering_wrapper(at::Tensor &D, float eps, float lam) {
    return GPU_DynamicalClustering(D.data_ptr<float>(), D.size(0), D.size(1), eps, lam);
}

std::vector<at::Tensor> EGG_SynC_wrapper(at::Tensor &D, float eps) {
    return EGG_SynC(D.data_ptr<float>(), D.size(0), D.size(1), eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{
m.def("GPU_DynamicalClustering_wrapper", &GPU_DynamicalClustering_wrapper, "");
m.def("EGG_SynC_wrapper", &EGG_SynC_wrapper, "");
m.def("DynamicalClustering", &DynamicalClustering, "");
m.def("DynamicalClustering_wrapper", &DynamicalClustering_wrapper, "");
m.def("DynamicalClustering_parallel_wrapper", &DynamicalClustering_parallel_wrapper, "");
m.def("FDynamicalClustering_wrapper", &FDynamicalClustering_wrapper, "");
};