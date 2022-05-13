//
// Created by jakobrj on 4/22/22.
//

#ifndef GPU_SYNC_SIMPLE_GPU_SYNC_CUH
#define GPU_SYNC_SIMPLE_GPU_SYNC_CUH

#include <vector>
#include <ATen/ATen.h>

std::vector <at::Tensor> GPU_DynamicalClustering(float *h_D, int n, int d, float eps, float lam);

#endif //GPU_SYNC_SIMPLE_GPU_SYNC_CUH
