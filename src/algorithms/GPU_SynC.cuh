//
// Created by mrjak on 20-07-2021.
//

#ifndef GPU_SYNC_GPU_SYNC_H
#define GPU_SYNC_GPU_SYNC_H

#include "SynC.h"

clustering GPU_DynamicalClustering(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID1(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID2(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID_2D1(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID_2D2(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID_STAD0(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID_STAD1(float *h_D, int n, int d, float eps, float lam);

clustering
GPU_DynamicalClustering_GRID_STAD2(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version);

clustering GPU_DynamicalClustering_list(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version);

clustering GPU_DynamicalClustering_GRID_STAD3(float *h_D, int n, int d, float eps, float lam);

clustering GPU_DynamicalClustering_GRID_STAD4(float *h_D, int n, int d, float eps, float lam, int cell_factor);

clustering GPU_DynamicalClustering_DOUBLE_GRID(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version);

#endif //GPU_SYNC_GPU_SYNC_H
