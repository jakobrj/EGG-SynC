//
// Created by mrjak on 14-07-2021.
//

#ifndef GPU_SYNC_SYNC_H
#define GPU_SYNC_SYNC_H

// #include <ATen/ATen.h>
// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <vector>

using clustering_outliers = std::vector<std::vector<std::vector<int>>>;
using clustering = std::vector<std::vector<int>>;
using outliers = std::vector<int>;
using neighborhood = std::vector<int>;
using cluster = std::vector<int>;

neighborhood ComputeNeighborhood(int p, float eps, float *D, int n, int d);

void UpdatePoint(int x, const neighborhood &N_p, float *D_current, float *D_next, int n, int d);

float ComputeLocationOrder(int x, const neighborhood &N_x, float *D, int n, int d);

clustering synCluster(float *D, int n, int d, float eps);

std::vector<int> Outliers(float *D, int n, int d, clustering &C);

float NN(float *D, int n, int d, int k);

float K(float x);

float f_hat(int x, float *D, int n, int d, float *h);

float L(float *D, int n, int d, clustering &M);

clustering DynamicalClustering(float *D, int n, int d, float eps, float lam);

clustering SynC(float *D, int n, int d, int k);

clustering FDynamicalClustering(float *D, int n, int d, float eps, float lam, int B);

#endif //GPU_SYNC_SYNC_H
