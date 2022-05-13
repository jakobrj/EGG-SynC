//
// Created by jakobrj on 4/22/22.
//

#ifndef GPU_SYNC_EGG_SYNC_CUH
#define GPU_SYNC_EGG_SYNC_CUH

#include <vector>
#include <ATen/ATen.h>

std::vector <at::Tensor> EGG_SynC(float *h_D, int n, int d, float eps);

#endif //GPU_SYNC_EGG_SYNC_CUH
