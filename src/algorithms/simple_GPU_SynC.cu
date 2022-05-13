//
// Created by jakobrj on 4/22/22.
//
#include "simple_GPU_SynC.cuh"
#include "../utils/GPU_utils.cuh"
#include "../utils/CPU_math.h"
#include "../utils/Timer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#define BLOCK_SIZE 128

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


__global__ void kernel_itr(float *d_r_local, float *d_D_current, float *d_D_next, int n, int d, float eps) {
//    float *sum = new float[d];
    extern __shared__ float s[];
    float *sum = &s[d * threadIdx.x];
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;
        for (int q = 0; q < n; q++) {

            float dist = 0.;
            for (int l = 0; l < d; l++) {
                float diff = d_D_current[p * d + l] - d_D_current[q * d + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (dist <= eps) {
                number_of_neighbors++;
                for (int l = 0; l < d; l++) {
                    sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                }
                r_c += exp(-abs(dist));
            }
        }
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }
        //        if (p == 0) {
        //            printf("GPU: r_c=%f/%d=%f\n", r_c, number_of_neighbors, r_c / number_of_neighbors);
        //        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
//    delete sum;
}


__global__ void
GPU_synCluster(int *__restrict__ d_C, int *__restrict__ d_incl, const float *__restrict__ d_D_current, const int n,
               const int d, const float eps) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        int min = p;
        int count = 0;
        for (int q = 0; q < n; q++) {
            float dist = 0.;
            for (int l = 0; l < d; l++) {
                float diff = d_D_current[p * d + l] - d_D_current[q * d + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist <= eps) {
                count++;
                if (q < min) {
                    min = q;
                }
            }
        }
        if (count > 1) {
            d_C[p] = min;
            d_incl[min] = 1;
        }
    }
}

__global__ void rename_cluster(int *__restrict__ d_C, int *__restrict__ d_map, const int n) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        if (d_C[p] >= 0)
            d_C[p] = d_map[d_C[p]] - 1;
    }
}

std::vector <at::Tensor> GPU_DynamicalClustering(float *h_D, int n, int d, float eps, float lam) {

    gpu_reset_max_memory_usage();
    Timer<6> timer;

    timer.start_stage_time(0);
    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.end_stage_time(0);
    while (r_local < lam && itr < 100) {
        timer.start_itr_time();

        timer.start_stage_time(2);
        gpu_set_all_zero(d_r_local, 1);
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());
        itr++;

        kernel_itr << < number_of_blocks, min(n, BLOCK_SIZE), d * BLOCK_SIZE * sizeof(float)>> > (
                d_r_local, d_D_current, d_D_next, n, d, eps
        );
//        cudaDeviceSynchronize();
//        gpuErrchk(cudaPeekAtLastError());

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.end_stage_time(2);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("simple GPU - itr: %d, r_local: %f\n", itr, r_local);

        timer.end_itr_time();
    }

//    clustering C;
    timer.start_stage_time(4);

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    inclusive_scan(d_incl, d_map, n);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    timer.end_stage_time(4);
    timer.start_stage_time(5);

    int *h_C = copy_D_to_H(d_C, n);
//
//    int k = maximum(h_C, n) + 1;
//    if (k > 0) {
//        for (int i = 0; i < k; i++) {
//            cluster c;
//            C.push_back(c);
//        }
//        for (int p = 0; p < n; p++) {
//            if (h_C[p] >= 0)
//                C[h_C[p]].push_back(p);
//        }
//    }

    at::Tensor C = at::zeros(n);
//    int k = maximum(h_C, n) + 1;
//    if (k > 0) {
//        for (int i = 0; i < k; i++) {
//            cluster c;
//            C.push_back(c);
//        }
//        for (int p = 0; p < n; p++) {
//            if (h_C[p] >= 0)
//                C[h_C[p]].push_back(p);
//        }
//    }
    for (int p = 0; p < n; p++) {
        C[p] = h_C[p];
    }

//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
    gpu_free(d_D_current);
    gpu_free(d_D_next);
    gpu_free(d_C);
    gpu_free(d_incl);
    gpu_free(d_map);
    delete h_C;
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());

    std::vector <at::Tensor> result;
    result.push_back(C);

    at::Tensor itr_times = timer.get_itr_times();
    result.push_back(itr_times);
    at::Tensor stage_times = timer.get_stage_times();
    result.push_back(stage_times);
    at::Tensor space = at::zeros(1, at::kInt);
    space[0] = (int) gpu_max_memory_usage();
    result.push_back(space);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.end_stage_time(5);
    return result;

//    return C;
}