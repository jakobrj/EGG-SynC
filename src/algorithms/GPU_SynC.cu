//
// Created by mrjak on 20-07-2021.
//

#include "GPU_SynC.cuh"
#include "../utils/GPU_utils.cuh"
#include "../utils/CPU_math.h"

#include <cuda.h>
#include <cuda_runtime.h>

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

__device__ int get_start(const int *d_array, const int idx) {
    return idx > 0 ? d_array[idx - 1] : 0;
}

__device__ int get_end(const int *d_array, const int idx) {
    return d_array[idx];
}

__device__ int get_start(const unsigned int *d_array, const int idx) {
    return idx > 0 ? d_array[idx - 1] : 0;
}

__device__ int get_end(const unsigned int *d_array, const int idx) {
    return d_array[idx];
}

void swap(unsigned int *&array1, unsigned int *&array2) {
    unsigned int *tmp = array1;
    array1 = array2;
    array2 = tmp;
}

void swap(int *&array1, int *&array2) {
    int *tmp = array1;
    array1 = array2;
    array2 = tmp;
}

void swap(float *&array1, float *&array2) {
    float *tmp = array1;
    array1 = array2;
    array2 = tmp;
}

__global__ void kernel_itr(float *d_r_local, float *d_D_current, float *d_D_next, int n, int d, float eps) {
    float *sum = new float[d];
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
    delete sum;
}

__global__ void kernel_UpdatePoints(const float *__restrict__ d_D_current, float *__restrict__ d_D_next,
                                    const int n, const int d, const float eps) {
    float *sum = new float[d];
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
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
            }
        }
        if (p == 0) {
            printf("GPU: %d \n", number_of_neighbors);
        }
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            if (p == 0) {
                printf("%f ", d_D_next[p * d + l]);
            }
        }
        if (p == 0) {
            printf("\n");
        }
    }
    delete sum;
}

__global__ void kernel_ComputeLocalOrder(float *__restrict__ d_r_local,
                                         float *__restrict__ d_D_next,
                                         const int n, const int d, const float eps) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        float r_c = 0.;
        int number_of_neighbors = 0;
        for (int q = 0; q < n; q++) {
            float dist = 0.;
            for (int l = 0; l < d; l++) {
                float diff = d_D_next[p * d + l] - d_D_next[q * d + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (dist <= eps) {
                number_of_neighbors++;
                r_c += exp(-abs(dist));
            }
        }

        if (p == 0)
            printf("GPU r_c: %f, %d, %f\n", r_c, number_of_neighbors, r_c / number_of_neighbors);
        if (number_of_neighbors == 0) { //todo this case does not happen p is also neighbor of it self
            r_c = 1.;
        } else {
            r_c /= number_of_neighbors;
        }

        atomicAdd(d_r_local, r_c);
    }
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

clustering GPU_DynamicalClustering(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);
    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        gpu_set_all_zero(d_r_local, 1);
        itr++;

        kernel_itr << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local, d_D_current, d_D_next, n, d, eps);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("simple GPU - itr: %d, r_local: %f\n", itr, r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    delete h_C;

    return C;
}

__global__ void
kernel_grid_sizes(int *d_grid_sizes, float *d_D_current, int n, int d, float cell_size, int width, int grid_dims) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int number_of_cells = 1;
        int cell = 0;
        for (int i = 0; i < grid_dims; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / cell_size;
            if (tmp == width)
                tmp--;
            cell += tmp * number_of_cells;
            number_of_cells *= width;
        }

        atomicInc((unsigned int *) &d_grid_sizes[cell], n);
    }
}

__global__ void
kernel_grid_populate(int *d_grid, int *d_grid_ends, int *d_grid_sizes, float *d_D_current, int n, int d,
                     float cell_size,
                     int width,
                     int grid_dims) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int number_of_cells = 1;
        int cell = 0;
        for (int i = 0; i < grid_dims; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / cell_size;
            if (tmp == width)
                tmp--;
            cell += tmp * number_of_cells;
            number_of_cells *= width;
        }

        int offset = get_start(d_grid_ends, cell);

        int idx = atomicInc((unsigned int *) &d_grid_sizes[cell], n);
        d_grid[offset + idx] = p;
    }
}

__global__ void
kernel_itr_grid_v1(float *d_r_local, int *d_grid, int *d_grid_ends, float *d_D_current, float *d_D_next, int n,
                   int d,
                   float eps, int width, int grid_dims) {
    float *sum = new float[d];

    int cs = 1;
    for (int i = 0; i < grid_dims; i++) {
        cs *= 3;
    }
    int *tmps = new int[cs];
    int *offsets = new int[grid_dims];

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < grid_dims; l++) {
            tmps[l] = -1;
        }

        for (int j = 0; j < cs; j++) {

            int number_of_cells = 1;
            offsets[j] = 0;
            for (int i = 0; i < grid_dims; i++) {
                float val = d_D_current[p * d + i];
                int tmp = val / eps;
                if (tmp == width)
                    tmp--;

                if (tmp + tmps[i] < 0 || tmp + tmps[i] >= width) {
                    offsets[j] = -1;
                    break;
                }

                offsets[j] += (tmp + tmps[i]) * number_of_cells;
                number_of_cells *= width;
            }

            int l = 0;
            while (l < grid_dims) {
                tmps[l]++;
                if (tmps[l] == 2) {
                    tmps[l] = -1;
                    l++;
                } else {
                    break;
                }
            }
        }

        //        int number_of_cells = 1;
        //        int cell = 0;
        //        for (int i = 0; i < grid_dims; i++) {
        //            float val = d_D_current[p * d + i];
        //            int tmp = val / eps;
        //            if (tmp == width)
        //                tmp--;
        //            cell += tmp * number_of_cells;
        //            number_of_cells *= width;
        //        }

        int cell_i = 0;
        while (offsets[cell_i] == -1 && cell_i < cs) {
            cell_i++;
        }
        int cell = offsets[cell_i];

        cell = cell > 0 ? cell - 1 : 0;

        int idx = get_start(d_grid_ends, cell);
        while (cell_i < cs) {

            int q = d_grid[idx];

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

            //update idx
            if (idx < d_grid_ends[cell]) {
                idx++;
            } else {
                cell_i++;
                while (offsets[cell_i] == -1 && cell_i < cs) {
                    cell_i++;
                }
                if (cell_i < cs) {
                    cell = offsets[cell_i];
                    idx = cell > 0 ? d_grid_ends[cell - 1] : 0;
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

__global__ void
kernel_itr_grid_v2(float *d_r_local, int *d_grid, int *d_grid_ends, float *d_D_current, float *d_D_next, int n,
                   int d,
                   float eps, int width, int grid_dims) {
    float *sum = new float[d];

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        int number_of_cells = 1;
        int cell_counter = 0;
        int cell = 0;
        for (int i = 0; i < grid_dims; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / eps;
            if (tmp == width)
                tmp--;
            cell += tmp * number_of_cells;
            number_of_cells *= width;
        }

        cell--;

        if (cell < 0) {
            cell++;
            cell_counter++;
        }

        int idx = get_start(d_grid_ends, cell);
        while (cell < width && cell_counter < number_of_cells) {

            int q = d_grid[idx];

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

            //update idx
            if (idx < d_grid_ends[cell]) {
                idx++;
            } else {
                cell++;
                cell_counter++;
                if (cell < width) {
                    idx = d_grid_ends[cell - 1];
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

__global__ void
kernel_itr_grid_v3(float *d_r_local, int *d_grid, int *d_grid_ends, float *d_D_current, float *d_D_next, int n,
                   int d,
                   float eps, int width, int grid_dims) {
    float *sum = new float[d];

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        int number_of_cells = 1;
        int cell_counter = 0;
        float val = d_D_current[p * d + 0];
        int tmp = val / eps;
        if (tmp == width)
            tmp--;
        int cell = tmp;

        cell--; //go to the left
        int start = get_start(d_grid_ends, cell);
        cell++; //go to the right
        cell++; //go to the right
        int end = cell < width ? d_grid_ends[cell] : n;
        for (int idx = start; idx < end; idx++) {

            int q = d_grid[idx];

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

        //        if(p == 0 ){
        //            printf("GPU GRID: r_c=%f/%d=%f\n", r_c, number_of_neighbors, r_c/number_of_neighbors);
        //        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

__global__ void
kernel_itr_grid(float *d_r_local, int *d_grid, int *d_grid_ends, float *d_D_current, float *d_D_next, int n, int d,
                float eps, int width, int grid_dims) {
    float *sum = new float[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {
        int p = d_grid[idx1];
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        int number_of_cells = 1;
        int cell_counter = 0;
        float val = d_D_current[p * d + 0];
        int tmp = val / eps;
        if (tmp == width)
            tmp--;
        int cell = tmp;

        cell--; //go to the left
        int start = cell > 0 ? d_grid_ends[cell - 1] : 0;
        cell++; //go to the right
        cell++; //go to the right
        int end = cell < width ? d_grid_ends[cell] : n;
        for (int idx = start; idx < end; idx++) {

            int q = d_grid[idx];

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
        //            printf("GPU GRID: r_c=%f/%d=%f\n", r_c, number_of_neighbors, r_c / number_of_neighbors);
        //        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

clustering GPU_DynamicalClustering_GRID1(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    int grid_dims = 1;
    int number_of_cells = 1;
    int width = ceil(1. / eps);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    while (r_local < lam) {
        gpu_set_all_zero(d_r_local, 1);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpu_set_all_zero(d_grid_ends, number_of_cells);

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, eps, width,
                grid_dims);

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, eps, width, grid_dims);

        kernel_itr_grid_v3 << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local, d_grid, d_grid_ends, d_D_current,
                d_D_next, n, d, eps, width, grid_dims);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    delete h_C;

    return C;
}

clustering GPU_DynamicalClustering_GRID2(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    int grid_dims = 1;
    int number_of_cells = 1;
    int width = ceil(1. / eps);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    while (r_local < lam) {
        gpu_set_all_zero(d_r_local, 1);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpu_set_all_zero(d_grid_ends, number_of_cells);

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, eps, width,
                grid_dims);

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, eps, width, grid_dims);

        kernel_itr_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                               (d_r_local, d_grid, d_grid_ends, d_D_current, d_D_next,
                                                       n, d, eps, width, grid_dims);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_2D1(float *d_r_local, int *d_grid, int *d_grid_ends, float *d_D_current, float *d_D_next, int n,
                    int d,
                    float eps, int width, int grid_dims, int number_of_cells) {
    float *sum = new float[d];

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        float val_0 = d_D_current[p * d + 0];
        int tmp_0 = val_0 / eps;
        if (tmp_0 == width)
            tmp_0--;
        float val_1 = d_D_current[p * d + 1];
        int tmp_1 = val_1 / eps;
        if (tmp_1 == width)
            tmp_1--;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (tmp_0 + i < 0 || tmp_1 + j < 0 || tmp_0 + i >= width || tmp_1 + j >= width) {
                    continue;
                }
                int cell = (tmp_0 + i) + (tmp_1 + j) * width;

                int start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                int end = cell < number_of_cells ? d_grid_ends[cell] : n;

                if (end - start == 0) {
                    continue;
                }

                //                if (p == 0)
                //                    printf("%d,%d\n", i, j);
                //                if (p == 0)
                //                    printf("start: %d, end: %d\n", start, end);
                for (int idx = start; idx < end; idx++) {

                    int q = d_grid[idx];

                    float dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float diff = d_D_current[p * d + l] - d_D_current[q * d + l];
                        dist += diff * diff;
                    }
                    dist = sqrt(dist);

                    if (dist <= eps) {

                        //                        if (p == 0) {
                        //                            printf("q: %d\n", q);
                        //                        }

                        number_of_neighbors++;
                        for (int l = 0; l < d; l++) {
                            sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                        }
                        r_c += exp(-abs(dist));
                    }
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        //        if (p == 0) {
        //            printf("GPU 2D1: r_c=%f/%d=%f\n", r_c, number_of_neighbors, r_c / number_of_neighbors);
        //        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

__global__ void
kernel_itr_grid_2D2(float *__restrict__ d_r_local, const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                    const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                    const float eps, const int width, const int grid_dims, const float cell_size,
                    int number_of_cells) {
    float *sum = new float[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        float val_0 = d_D_current[p * d + 0];
        int tmp_0 = val_0 / cell_size;
        if (tmp_0 == width)
            tmp_0--;
        float val_1 = d_D_current[p * d + 1];
        int tmp_1 = val_1 / cell_size;
        if (tmp_1 == width)
            tmp_1--;

        int radius = ceil(eps / cell_size);

        int i = -radius;
        int j = -radius - 1;
        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        while (true) {

            if (idx >= end) {
                int size;
                do {
                    j++;
                    if (j > radius) {
                        j = -radius;
                        i++;
                    }

                    size = 0;
                    if (tmp_0 + i >= 0 && tmp_1 + j >= 0 && tmp_0 + i < width && tmp_1 + j < width) {
                        cell = (tmp_0 + i) + (tmp_1 + j) * width;
                        start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                        end = cell < number_of_cells ? d_grid_ends[cell] : n;
                        size = end - start;
                    }
                } while (size == 0 && i <= radius);

                if (i > radius) {
                    break;
                }
                //                if (p == 0)
                //                    printf("%d,%d\n", i, j);
                //                if (p == 0) {
                //                    printf("start: %d, end: %d\n", start, end);
                //                }

                idx = start;
            }

            int q = d_grid[idx];

            float dist = 0.;
            for (int l = 0; l < d; l++) {
                float diff = d_D_current[p * d + l] - d_D_current[q * d + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (dist <= eps) {

                //                if (p == 0) {
                //                    printf("q: %d\n", q);
                //                }

                number_of_neighbors++;
                for (int l = 0; l < d; l++) {
                    sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                }
                r_c += exp(-abs(dist));
            }

            idx++;
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        if (p == 0) {
            printf("GPU 2D2: r_c=%f/%d=%f\n", r_c, number_of_neighbors, r_c / number_of_neighbors);
        }
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

clustering GPU_DynamicalClustering_GRID_2D1(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    int grid_dims = 2;
    int number_of_cells = 1;
    int width = ceil(1. / eps);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    while (r_local < lam) {
        gpu_set_all_zero(d_r_local, 1);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpu_set_all_zero(d_grid_ends, number_of_cells);

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, eps, width,
                grid_dims);

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, eps, width, grid_dims);

        kernel_itr_grid_2D1 << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local, d_grid, d_grid_ends, d_D_current,
                d_D_next, n, d, eps,
                width, grid_dims, number_of_cells);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    delete h_C;

    return C;
}

clustering GPU_DynamicalClustering_GRID_2D2(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int grid_dims = 2;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    while (r_local < lam) {
        gpu_set_all_zero(d_r_local, 1);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpu_set_all_zero(d_grid_ends, number_of_cells);

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        kernel_itr_grid_2D2 << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local, d_grid, d_grid_ends, d_D_current,
                d_D_next, n, d, eps,
                width, grid_dims, cell_size, number_of_cells);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD0(float *__restrict__ d_r_local, const int *__restrict__ d_grid,
                      const int *__restrict__ d_grid_ends,
                      const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                      const float eps, const int width, const int grid_dims, const float cell_size,
                      int number_of_cells) {
    float *sum = new float[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        float val_0 = d_D_current[p * d + 0];
        int tmp_0 = val_0 / cell_size;
        if (tmp_0 == width)
            tmp_0--;
        float val_1 = d_D_current[p * d + 1];
        int tmp_1 = val_1 / cell_size;
        if (tmp_1 == width)
            tmp_1--;

        int radius = ceil(eps / cell_size);

        int i = -radius;
        int j = -radius - 1;
        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        while (true) {

            if (idx >= end) {
                int size;
                do {
                    j++;
                    if (j > radius) {
                        j = -radius;
                        i++;
                    }

                    bool included = false;

                    float small_coor_0 = d_D_current[p * d + 0];
                    float small_coor_1 = d_D_current[p * d + 1];

                    if (i > 0) {
                        small_coor_0 = (tmp_0 + i) * cell_size;
                    } else if (i < 0) {
                        small_coor_0 = (tmp_0 + i + 1) * cell_size;
                    } else {
                        if ((d_D_current[p * d + 0] - ((tmp_0) * cell_size)) >
                            (((tmp_0 + 1) * cell_size) - d_D_current[p * d + 0])) {
                            small_coor_0 = (tmp_0 + i) * cell_size;
                        } else {
                            small_coor_0 = (tmp_0 + i + 1) * cell_size;
                        }
                    }

                    if (j > 0) {
                        small_coor_1 = (tmp_1 + j) * cell_size;
                    } else if (j < 0) {
                        small_coor_1 = (tmp_1 + j + 1) * cell_size;
                    } else {
                        if ((d_D_current[p * d + 1] - ((tmp_1) * cell_size)) >
                            (((tmp_1 + 1) * cell_size) - d_D_current[p * d + 1])) {
                            small_coor_1 = (tmp_1 + j) * cell_size;
                        } else {
                            small_coor_1 = (tmp_1 + j + 1) * cell_size;
                        }
                    }

                    float small_diff_0 = small_coor_0 - d_D_current[p * d + 0];
                    float small_diff_1 = small_coor_1 - d_D_current[p * d + 1];

                    float small_dist = sqrt(small_diff_0 * small_diff_0 + small_diff_1 * small_diff_1);
                    if (small_dist <= eps) {
                        included = true;
                    }

                    size = 0;
                    if (included && (tmp_0 + i >= 0 && tmp_1 + j >= 0 && tmp_0 + i < width && tmp_1 + j < width)) {
                        cell = (tmp_0 + i) + (tmp_1 + j) * width;
                        start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                        end = cell < number_of_cells ? d_grid_ends[cell] : n;
                        size = end - start;
                    }
                } while (size == 0 && i <= radius);

                if (i > radius) {
                    break;
                }

                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        float sum_delta_y_i = 0;
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            sum_delta_y_i += abs(sum[l]);
        }
        r_c += number_of_neighbors - sum_delta_y_i;
        r_c /= number_of_neighbors;
        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
}

__global__ void
GPU_synCluster_grid(int *__restrict__ d_C, int *__restrict__ d_incl, const float *__restrict__ d_D_current,
                    const int *__restrict__ d_grid_ends, const int *__restrict__ d_grid,
                    const int n, const int d, const float cell_size, const int grid_dims, const int width) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int number_of_cells = 1;
        int cell = 0;
        for (int i = 0; i < grid_dims; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / cell_size;
            if (tmp == width)
                tmp--;
            cell += tmp * number_of_cells;
            number_of_cells *= width;
        }

        int start = cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = d_grid_ends[cell];

        int min = d_grid[start];
        int count = end - start;

        if (count > 1) {
            d_C[p] = min;
            d_incl[min] = 1;
        }
    }
}

clustering GPU_DynamicalClustering_GRID_STAD0(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int grid_dims = 2;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    while (r_local < lam) {
        gpu_set_all_zero(d_r_local, 1);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpu_set_all_zero(d_grid_ends, number_of_cells);

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        kernel_itr_grid_STAD0 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                     (d_r_local, d_grid, d_grid_ends, d_D_current,
                                                             d_D_next, n, d, eps,
                                                             width, grid_dims, cell_size, number_of_cells);

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    //    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    //    gpu_set_all_zero(d_grid_ends, number_of_cells);
    //
    //    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                             (d_grid_sizes, d_D_current, n, d, cell_size, width, grid_dims);
    //
    //    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);
    //
    //    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    //
    //    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d, cell_size, width, grid_dims);
    //
    //
    //    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d, cell_size, grid_dims, width);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD1_v1(float *__restrict__ d_sum_cos, float *__restrict__ d_sum_sin, float *__restrict__ d_r_local,
                         const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                         const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n,
                         const int d,
                         const float eps, const int width, const int grid_dims, const float cell_size,
                         int number_of_cells) {

    float *sum = new float[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        float val_0 = d_D_current[p * d + 0];
        int tmp_0 = val_0 / cell_size;
        if (tmp_0 == width)
            tmp_0--;
        float val_1 = d_D_current[p * d + 1];
        int tmp_1 = val_1 / cell_size;
        if (tmp_1 == width)
            tmp_1--;

        int radius = ceil(eps / cell_size);

        int i = -radius;
        int j = -radius - 1;
        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {
                    j++;
                    if (j > radius) {
                        j = -radius;
                        i++;
                    }
                    if (i > radius) {
                        continue;
                    }

                    bool included = false;
                    fully_included = false;

                    float small_coor_0 = d_D_current[p * d + 0];
                    float small_coor_1 = d_D_current[p * d + 1];
                    float large_coor_0 = d_D_current[p * d + 0];
                    float large_coor_1 = d_D_current[p * d + 1];

                    if (i > 0) {
                        small_coor_0 = (tmp_0 + i) * cell_size;
                        large_coor_0 = (tmp_0 + i + 1) * cell_size;
                    } else if (i < 0) {
                        small_coor_0 = (tmp_0 + i + 1) * cell_size;
                        large_coor_0 = (tmp_0 + i) * cell_size;
                    } else {
                        if ((d_D_current[p * d + 0] - ((tmp_0) * cell_size)) >
                            (((tmp_0 + 1) * cell_size) - d_D_current[p * d + 0])) {
                            small_coor_0 = (tmp_0 + i) * cell_size;
                            large_coor_0 = (tmp_0 + i + 1) * cell_size;
                        } else {
                            small_coor_0 = (tmp_0 + i + 1) * cell_size;
                            large_coor_0 = (tmp_0 + i) * cell_size;
                        }
                    }

                    if (j > 0) {
                        small_coor_1 = (tmp_1 + j) * cell_size;
                        large_coor_1 = (tmp_1 + j + 1) * cell_size;
                    } else if (j < 0) {
                        small_coor_1 = (tmp_1 + j + 1) * cell_size;
                        large_coor_1 = (tmp_1 + j) * cell_size;
                    } else {
                        if ((d_D_current[p * d + 1] - ((tmp_1) * cell_size)) >
                            (((tmp_1 + 1) * cell_size) - d_D_current[p * d + 1])) {
                            small_coor_1 = (tmp_1 + j) * cell_size;
                            large_coor_1 = (tmp_1 + j + 1) * cell_size;
                        } else {
                            small_coor_1 = (tmp_1 + j + 1) * cell_size;
                            large_coor_1 = (tmp_1 + j) * cell_size;
                        }
                    }

                    const float small_diff_0 = small_coor_0 - d_D_current[p * d + 0];
                    const float small_diff_1 = small_coor_1 - d_D_current[p * d + 1];

                    const float small_dist = sqrt(small_diff_0 * small_diff_0 + small_diff_1 * small_diff_1);
                    if (small_dist <= eps) {
                        included = true;
                    }

                    const float large_diff_0 = large_coor_0 - d_D_current[p * d + 0];
                    const float large_diff_1 = large_coor_1 - d_D_current[p * d + 1];

                    const float large_dist = sqrt(large_diff_0 * large_diff_0 + large_diff_1 * large_diff_1);
                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    size = 0;
                    if (included && (tmp_0 + i >= 0 && tmp_1 + j >= 0 && tmp_0 + i < width && tmp_1 + j < width)) {
                        cell = (tmp_0 + i) + (tmp_1 + j) * width;
                        start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                        end = cell < number_of_cells ? d_grid_ends[cell] : n;
                        size = end - start;

                        if (fully_included && size > 0) {
                            for (int l = 0; l < d; l++) {
                                const float x_t = d_D_current[p * d + l];
                                const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                                           (d_sum_cos[cell * d + l] * sin(x_t)));
                                sum[l] += to_be_added;
                            }
                            number_of_neighbors += size;
                            size = 0; //todo
                        }
                    }
                } while (size == 0 && i <= radius);

                if (i > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        float sum_delta_y_i = 0;
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            sum_delta_y_i += abs(sum[l]);
        }
        r_c = number_of_neighbors - sum_delta_y_i;
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
    //    delete test_sum;
}

__global__ void
kernel_itr_grid_STAD1(float *__restrict__ d_sum_cos, float *__restrict__ d_sum_sin, float *__restrict__ d_r_local,
                      const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                      const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                      const float eps, const int width, const int grid_dims, const float cell_size,
                      int number_of_cells) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {

                    idxs[0]++;

                    {
                        int l = 1;
                        while (l < d && idxs[l - 1] > radius) {
                            idxs[l - 1] = -radius;
                            idxs[l]++;
                            l++;
                        }
                    }

                    if (idxs[d - 1] > radius) {
                        continue;
                    }

                    bool included = false;
                    fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    bool within_bounds = true;
                    for (int l = 0; l < d; l++) {
                        if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                            within_bounds = false;
                        }
                    }

                    size = 0;
                    if (included && within_bounds) {
                        number_of_cells = 1;
                        cell = 0;
                        for (int l = 0; l < d; l++) {
                            cell += (tmp[l] + idxs[l]) * number_of_cells;
                            number_of_cells *= width;
                        }

                        if (cell >= number_of_cells) {
                            printf("cell: %d\n", cell);
                        }
                        start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                        end = cell < number_of_cells ? d_grid_ends[cell] : n;
                        size = end - start;

                        if (fully_included && size > 0) {
                            for (int l = 0; l < d; l++) {
                                const float x_t = d_D_current[p * d + l];
                                const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                                           (d_sum_cos[cell * d + l] * sin(x_t)));
                                sum[l] += to_be_added;
                            }
                            number_of_neighbors += size;
                            size = 0;
                        }
                    }
                } while (size == 0 && idxs[d - 1] <= radius);

                if (idxs[d - 1] > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        float sum_delta_y_i = 0;
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            sum_delta_y_i += abs(sum[l]);
        }
        r_c = number_of_neighbors - sum_delta_y_i;
        r_c /= number_of_neighbors;

        atomicAdd(d_r_local, r_c); //todo must be initialized before the kernel call
    }
    delete sum;
    delete tmp;
    delete idxs;
}

__global__ void
kernel_STAD(float *d_sum_cos, float *d_sum_sin, float *d_D_current, int n, int d, float cell_size, int width,
            int grid_dims) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int number_of_cells = 1;
        int cell = 0;
        for (int i = 0; i < grid_dims; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / cell_size;
            if (tmp == width)
                tmp--;
            cell += tmp * number_of_cells;
            number_of_cells *= width;
        }

        for (int i = 0; i < d; i++) {
            float val = d_D_current[p * d + i];
            atomicAdd(&d_sum_cos[cell * d + i], cos(val));
            atomicAdd(&d_sum_sin[cell * d + i], sin(val));
        }
    }
}

clustering GPU_DynamicalClustering_GRID_STAD1(float *h_D, int n, int d, float eps, float lam) {

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);
    float *d_sum_cos = gpu_malloc_float(number_of_cells * d);
    float *d_sum_sin = gpu_malloc_float(number_of_cells * d);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all_zero(d_r_local, 1);
        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_grid_ends, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_cos, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_sin, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_STAD << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                           (d_sum_cos, d_sum_sin, d_D_current, n, d, cell_size, width,
                                                   grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_itr_grid_STAD1 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                     //        kernel_itr_grid_STAD1 << < 1, 1 >> >
                                                     (d_sum_cos, d_sum_sin, d_r_local, d_grid, d_grid_ends, d_D_current, d_D_next, n, d, eps,
                                                             width, grid_dims, cell_size, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1) / n;
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    //    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    //    gpu_set_all_zero(d_grid_ends, number_of_cells);
    //
    //    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                             (d_grid_sizes, d_D_current, n, d, cell_size, width, grid_dims);
    //
    //    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);
    //
    //    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    //
    //    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d, cell_size, width, grid_dims);
    //
    //
    //    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
    //                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d, cell_size, grid_dims, width);

    GPU_synCluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_incl, d_D_current, n, d, eps);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);

    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD2(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
                      float *__restrict__ d_r_local,
                      const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                      const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                      const float eps, const int width, const int grid_dims, const float cell_size,
                      int number_of_cells, int itr) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {

                    ///increment index
                    idxs[0]++;
                    int l = 1;
                    while (l < d && idxs[l - 1] > radius) {
                        idxs[l - 1] = -radius;
                        idxs[l]++;
                        l++;
                    }
                    if (idxs[d - 1] > radius) {
                        continue;
                    }

                    ///check within bounds
                    bool within_bounds = true;
                    for (int l = 0; l < d; l++) {
                        if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                            within_bounds = false;
                        }
                    }
                    if (!within_bounds) {
                        continue;
                    }

                    ///check size
                    number_of_cells = 1;
                    cell = 0;
                    for (int l = 0; l < d; l++) {
                        cell += (tmp[l] + idxs[l]) * number_of_cells;
                        number_of_cells *= width;
                    }

                    start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                    end = cell < number_of_cells ? d_grid_ends[cell] : n;
                    size = end - start;
                    if (size == 0) {
                        continue;
                    }

                    ///compute included
                    bool included = false;
                    fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    if (fully_included) {
                        for (int l = 0; l < d; l++) {
                            const float x_t = d_D_current[p * d + l];
                            const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                                       (d_sum_cos[cell * d + l] * sin(x_t)));
                            sum[l] += to_be_added;
                        }
                        number_of_neighbors += size;
                        size = 0;
                    } else if (!included) {
                        size = 0;
                    }

                } while (size == 0 && idxs[d - 1] <= radius);

                if (idxs[d - 1] > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        number_of_cells = 1;
        cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell) {
            //            if (itr > 10) {
            //                printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                       number_of_points_in_cell);
            //            }
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

__global__ void
kernel_itr_grid_STAD2_1(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
                        float *__restrict__ d_r_local,
                        const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                        const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                        const float eps, const int width, const int grid_dims, const float cell_size,
                        int number_of_cells, int itr) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = idx1;

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {

                    ///increment index
                    idxs[0]++;
                    int l = 1;
                    while (l < d && idxs[l - 1] > radius) {
                        idxs[l - 1] = -radius;
                        idxs[l]++;
                        l++;
                    }
                    if (idxs[d - 1] > radius) {
                        continue;
                    }

                    ///check within bounds
                    bool within_bounds = true;
                    for (int l = 0; l < d; l++) {
                        if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                            within_bounds = false;
                        }
                    }
                    if (!within_bounds) {
                        continue;
                    }

                    ///check size
                    number_of_cells = 1;
                    cell = 0;
                    for (int l = 0; l < d; l++) {
                        cell += (tmp[l] + idxs[l]) * number_of_cells;
                        number_of_cells *= width;
                    }

                    start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                    end = cell < number_of_cells ? d_grid_ends[cell] : n;
                    size = end - start;
                    if (size == 0) {
                        continue;
                    }

                    ///compute included
                    bool included = false;
                    fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    if (fully_included) {
                        for (int l = 0; l < d; l++) {
                            const float x_t = d_D_current[p * d + l];
                            const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                                       (d_sum_cos[cell * d + l] * sin(x_t)));
                            sum[l] += to_be_added;
                        }
                        number_of_neighbors += size;
                        size = 0;
                    } else if (!included) {
                        size = 0;
                    }

                } while (size == 0 && idxs[d - 1] <= radius);

                if (idxs[d - 1] > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        number_of_cells = 1;
        cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell) {
            //            if (itr > 10) {
            //                printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                       number_of_points_in_cell);
            //            }
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

__global__ void
kernel_itr_grid_STAD2_2(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
                        float *__restrict__ d_r_local,
                        const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                        const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n, const int d,
                        const float eps, const int width, const int grid_dims, const float cell_size,
                        int number_of_cells, int itr) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        for (int l = 0; l < d; l++) {
            idxs[l] = -radius;
        }

        while (idxs[d - 1] <= radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < d; l++) {
                if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }
            if (within_bounds) {

                ///check size
                number_of_cells = 1;
                int cell = 0;
                for (int l = 0; l < d; l++) {
                    cell += (tmp[l] + idxs[l]) * number_of_cells;
                    number_of_cells *= width;
                }

                int cell_start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                int cell_end = cell < number_of_cells ? d_grid_ends[cell] : n;
                int size = cell_end - cell_start;
                if (size > 0) {

                    ///compute included
                    bool included = false;
                    bool fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    if (included) {
                        if (fully_included) {
                            for (int l = 0; l < d; l++) {
                                const float x_t = d_D_current[p * d + l];
                                const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                                           (d_sum_cos[cell * d + l] * sin(x_t)));
                                sum[l] += to_be_added;
                            }
                            number_of_neighbors += size;
                        } else {
                            for (int q_idx = cell_start; q_idx < cell_end; q_idx++) {
                                int q = d_grid[q_idx];

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
                                }
                            }
                        }
                    }
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < d && idxs[l - 1] > radius) {
                idxs[l - 1] = -radius;
                idxs[l]++;
                l++;
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        number_of_cells = 1;
        int cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell) {
            //            if (itr > 10) {
            //                printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                       number_of_points_in_cell);
            //            }
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

/// change termination criteria
clustering
GPU_DynamicalClustering_GRID_STAD2(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d) / cell_factor;
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);
    float *d_sum_cos = gpu_malloc_float(number_of_cells * d);
    float *d_sum_sin = gpu_malloc_float(number_of_cells * d);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all(d_r_local, 1, 1.);
        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_grid_ends, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_cos, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_sin, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_STAD << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                           (d_sum_cos, d_sum_sin, d_D_current, n, d, cell_size, width,
                                                   grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        //        if (itr > 0) {
        //            for (int i = 0; i < width; i++) {
        //                print_array_nonzero_gpu(&d_grid_sizes[i * width], width);
        //                printf("\n");
        //            }
        //        }

        if (version == 1) {

            kernel_itr_grid_STAD2_1 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                           (d_sum_cos, d_sum_sin, d_r_local, d_grid,
                                                                   d_grid_ends, d_D_current, d_D_next, n, d,
                                                                   eps,
                                                                   width, grid_dims, cell_size,
                                                                   number_of_cells, itr);
        } else if (version == 2) {

            kernel_itr_grid_STAD2_2 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                           (d_sum_cos, d_sum_sin, d_r_local, d_grid,
                                                                   d_grid_ends, d_D_current, d_D_next, n, d,
                                                                   eps,
                                                                   width, grid_dims, cell_size,
                                                                   number_of_cells, itr);
        } else {

            kernel_itr_grid_STAD2 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                         (d_sum_cos, d_sum_sin, d_r_local, d_grid,
                                                                 d_grid_ends, d_D_current, d_D_next, n, d,
                                                                 eps,
                                                                 width, grid_dims, cell_size,
                                                                 number_of_cells, itr);
        }

        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1);
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    gpu_set_all_zero(d_grid_ends, number_of_cells);

    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, cell_size, width,
            grid_dims);

    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);

    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d,
                                                        cell_size, width, grid_dims);

    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d,
                                                       cell_size, grid_dims, width);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);

    delete h_C;

    return C;
}

__device__ int compute_cell_id(const float *d_D, const int p, const int d, const int width, const int grid_dims,
                               const float cell_size) {
    int number_of_cells = 1;
    int cell = 0;
    for (int i = 0; i < grid_dims; i++) {
        float val = d_D[p * d + i];
        int tmp = val / cell_size;
        if (tmp == width)
            tmp--;
        cell += tmp * number_of_cells;
        number_of_cells *= width;
    }
    return cell;
}

__device__ bool compute_cell_matches(const float *d_D, const unsigned int *d_list_cell_dim_ids,
                                     int cell_idx, int p, int d, int width, float cell_size) {
    for (int i = 0; i < d; i++) {
        float val = d_D[p * d + i];
        int tmp = val / cell_size;
        if (tmp == width)
            tmp--;
        if (d_list_cell_dim_ids[cell_idx * d + i] != tmp) {
            return false;
        }
    }
    return true;
}

__device__ int compute_cell_idx(const float *d_D, const unsigned int *d_list_cell_dim_ids, int p, int d, int width,
                                float cell_size) {
    int cell_idx = 0;
    while (!compute_cell_matches(d_D, d_list_cell_dim_ids, cell_idx, p, d, width, cell_size)) {
        cell_idx++;
    }
    return cell_idx;
}

__global__ void
kernel_write_list_cell_id(unsigned int *d_list_cell_dim_ids, float *d_D, int n, int d,
                          int width, float cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int i = 0; i < d; i++) {
            float val = d_D[p * d + i];
            int tmp = val / cell_size;
            if (tmp == width)
                tmp--;
            d_list_cell_dim_ids[p * d + i] = tmp;
        }
    }
}

__global__ void
kernel_compute_list_cell_size(unsigned int *d_list_cell_sizes, unsigned int *d_list_cell_included,
                              unsigned int *d_list_cell_dim_ids, float *d_D,
                              int n, int d, int width, float cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        int cell_idx = compute_cell_idx(d_D, d_list_cell_dim_ids, p, d, width, cell_size);
        int tmp1 = d_D[p * d + 0] / cell_size;
        int tmp2 = d_D[p * d + 1] / cell_size;
        atomicInc(&d_list_cell_sizes[cell_idx], n);
        d_list_cell_included[cell_idx] = 1;
    }
}

__global__ void kernel_repack_list_cells(unsigned int *d_new_list_cell_ends, unsigned int *d_new_list_cell_dim_ids,
                                         unsigned int *d_list_cell_included, unsigned int *d_list_cell_idxs,
                                         unsigned int *d_list_cell_ends, unsigned int *d_list_cell_dim_ids,
                                         int n, int d) {
    for (int old_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         old_cell_idx < n; old_cell_idx += blockDim.x * gridDim.x) {

        if (d_list_cell_included[old_cell_idx] > 0) {

            int new_cell_idx = get_end(d_list_cell_idxs, old_cell_idx) - 1;
            int cell_end = get_end(d_list_cell_ends, old_cell_idx);

            d_new_list_cell_ends[new_cell_idx] = cell_end;
            //            d_new_list_cell_ids[new_cell_idx] = d_inner_grid_ids[old_cell_idx];

            for (int i = 0; i < d; i++) {
                d_new_list_cell_dim_ids[new_cell_idx * d + i] = d_list_cell_dim_ids[old_cell_idx * d + i];
            }
        }
    }
}

__global__ void kernel_populate_list_cell(float *d_sum_cos, float *d_sum_sin, unsigned int *d_list_cell_points,
                                          unsigned int *d_list_cell_ends, unsigned int *d_list_cell_sizes,
                                          unsigned int *d_list_cell_dim_ids, float *d_D,
                                          int n, int d, int width, float cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        //        int cell_id = compute_cell_id(d_D, p, d, width, grid_dims, cell_size);
        int cell_idx = compute_cell_idx(d_D, d_list_cell_dim_ids, p, d, width, cell_size);

        int cell_start = get_start(d_list_cell_ends, cell_idx);
        int point_idx = atomicInc(&d_list_cell_sizes[cell_idx], n);
        d_list_cell_points[cell_start + point_idx] = p;

        for (int i = 0; i < d; i++) {
            float val = d_D[p * d + i];
            atomicAdd(&d_sum_cos[cell_idx * d + i], cos(val));
            atomicAdd(&d_sum_sin[cell_idx * d + i], sin(val));
        }
    }
}

__device__ float gpu_distance(int p, int q, const float *d_D, int d) {

    float dist = 0.;
    for (int l = 0; l < d; l++) {
        float diff = d_D[p * d + l] - d_D[q * d + l];
        dist += diff * diff;
    }
    dist = sqrt(dist);

    return dist;
}

__global__ void kernel_itr_list(float *d_D_next, float *d_r_local,
                                unsigned int *d_list_cell_ends, unsigned int *d_list_cell_dim_ids,
                                unsigned int *d_list_cell_points,
                                float *d_sum_sin, float *d_sum_cos, float *d_D_current,
                                int n, int d, int grid_number_of_cells, float eps, int width, float cell_size) {

    float *sum = new float[d];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        int p = d_list_cell_points[p_idx];

        int number_of_neighbors = 0;
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }

        for (int cell_idx = 0; cell_idx < grid_number_of_cells; cell_idx++) {

            int cell_start = get_start(d_list_cell_ends, cell_idx);
            int cell_end = get_end(d_list_cell_ends, cell_idx);
            int cell_number_of_points = cell_end - cell_start;

            ///compute included
            bool included = false;
            bool fully_included = false;

            float small_dist = 0.;
            float large_dist = 0.;
            for (int l = 0; l < d; l++) {
                float left_coor = d_list_cell_dim_ids[cell_idx * d + l] * cell_size;
                float right_coor = (d_list_cell_dim_ids[cell_idx * d + l] + 1) * cell_size;
                left_coor -= d_D_current[p * d + l];
                right_coor -= d_D_current[p * d + l];
                left_coor *= left_coor;
                right_coor *= right_coor;
                if (left_coor < right_coor) {
                    small_dist += left_coor;
                    large_dist += right_coor;
                } else {
                    small_dist += right_coor;
                    large_dist += left_coor;
                }
            }
            small_dist = sqrt(small_dist);
            large_dist = sqrt(large_dist);

            if (small_dist <= eps) {
                included = true;
            }

            if (large_dist <= eps) {
                fully_included = true;
            }

            if (fully_included) {
                for (int l = 0; l < d; l++) {
                    const float x_t = d_D_current[p * d + l];
                    const float to_be_added = ((d_sum_sin[cell_idx * d + l] * cos(x_t)) -
                                               (d_sum_cos[cell_idx * d + l] * sin(x_t)));
                    sum[l] += to_be_added;
                }
                number_of_neighbors += cell_number_of_points;
            } else if (included) {
                for (int q_idx = cell_start; q_idx < cell_end; q_idx++) {
                    int q = d_list_cell_points[q_idx];

                    if (gpu_distance(p, q, d_D_current, d) <= eps) {
                        for (int l = 0; l < d; l++) {
                            sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                        }
                        number_of_neighbors++;
                    }
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        int cell_idx = compute_cell_idx(d_D_current, d_list_cell_dim_ids, p, d, width, cell_size);
        int number_of_points_in_cell = get_end(d_list_cell_ends, cell_idx) - get_start(d_list_cell_ends, cell_idx);
        if (number_of_neighbors != number_of_points_in_cell) {
            d_r_local[0] = 0.;
        }
    }

    delete sum;
}

__device__ void
find_next_idx(int &q_idx, int &cell_idx, int &cell_end, int &number_of_neighbors, unsigned int *d_list_cell_ends,
              unsigned int *d_list_cell_dim_ids, float *d_D_current, float *d_sum_sin, float *d_sum_cos,
              float *sum,
              int p, int d, int grid_number_of_cells, float cell_size, float eps) {
    q_idx++;
    if (q_idx >= cell_end) {
        cell_idx++;
    }
    while (q_idx >= cell_end && cell_idx < grid_number_of_cells) {
        int cell_start = get_start(d_list_cell_ends, cell_idx);
        cell_end = get_end(d_list_cell_ends, cell_idx);
        int cell_number_of_points = cell_end - cell_start;
        q_idx = cell_start;

        bool included = false;
        bool fully_included = false;

        float small_dist = 0.;
        float large_dist = 0.;
        for (int l = 0; l < d; l++) {
            float left_coor = d_list_cell_dim_ids[cell_idx * d + l] * cell_size;
            float right_coor = (d_list_cell_dim_ids[cell_idx * d + l] + 1) * cell_size;
            left_coor -= d_D_current[p * d + l];
            right_coor -= d_D_current[p * d + l];
            left_coor *= left_coor;
            right_coor *= right_coor;
            if (left_coor < right_coor) {
                small_dist += left_coor;
                large_dist += right_coor;
            } else {
                small_dist += right_coor;
                large_dist += left_coor;
            }
        }
        small_dist = sqrt(small_dist);
        large_dist = sqrt(large_dist);

        if (small_dist <= eps) {
            included = true;
        }

        if (large_dist <= eps) {
            fully_included = true;
        }

        if (fully_included) {
            for (int l = 0; l < d; l++) {
                const float x_t = d_D_current[p * d + l];
                const float to_be_added = ((d_sum_sin[cell_idx * d + l] * cos(x_t)) -
                                           (d_sum_cos[cell_idx * d + l] * sin(x_t)));
                sum[l] += to_be_added;
            }
            number_of_neighbors += cell_number_of_points;
            q_idx += cell_number_of_points;
            cell_idx++;
        } else if (!included) {
            q_idx += cell_number_of_points;
            cell_idx++;
        }
    }
}

__global__ void kernel_itr_list_2(float *d_D_next, float *d_r_local,
                                  unsigned int *d_list_cell_ends, unsigned int *d_list_cell_dim_ids,
                                  unsigned int *d_list_cell_points,
                                  float *d_sum_sin, float *d_sum_cos, float *d_D_current,
                                  int n, int d, int grid_number_of_cells, float eps, int width, float cell_size) {

    float *sum = new float[d];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        int p = d_list_cell_points[p_idx];

        int number_of_neighbors = 0;
        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }

        int cell_end = -1;
        int cell_idx = -1;
        int q_idx = -1;

        find_next_idx(q_idx, cell_idx, cell_end, number_of_neighbors, d_list_cell_ends,
                      d_list_cell_dim_ids, d_D_current, d_sum_sin, d_sum_cos, sum,
                      p, d, grid_number_of_cells, cell_size, eps);

        while (cell_idx < grid_number_of_cells) {
            int q = d_list_cell_points[q_idx];

            if (gpu_distance(p, q, d_D_current, d) <= eps) {
                for (int l = 0; l < d; l++) {
                    sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                }
                number_of_neighbors++;
            }

            find_next_idx(q_idx, cell_idx, cell_end, number_of_neighbors, d_list_cell_ends,
                          d_list_cell_dim_ids, d_D_current, d_sum_sin, d_sum_cos, sum,
                          p, d, grid_number_of_cells, cell_size, eps);
        }

        if (number_of_neighbors == 0) {
            printf("p=%d, number_of_neighbors=%d\n", p, number_of_neighbors);
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        int center_cell_idx = compute_cell_idx(d_D_current, d_list_cell_dim_ids, p, d, width, cell_size);
        int number_of_points_in_cell =
                get_end(d_list_cell_ends, center_cell_idx) - get_start(d_list_cell_ends, center_cell_idx);
        if (number_of_neighbors != number_of_points_in_cell) {
            d_r_local[0] = 0.;
        }
    }

    delete sum;
}

void build_list_grid(float *&d_D_current,
                     unsigned int *&d_list_cell_dim_ids, unsigned int *&d_new_list_cell_dim_ids,
                     unsigned int *&d_list_cell_included, unsigned int *&d_list_cell_idxs,
                     unsigned int *&d_list_cell_sizes, unsigned int *&d_list_cell_ends,
                     unsigned int *&d_new_list_cell_ends, unsigned int *&d_list_cell_points,
                     float *d_sum_cos, float *d_sum_sin,
                     int n, int d, int width, float cell_size) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    gpu_set_all_zero(d_list_cell_sizes, n);
    gpu_set_all_zero(d_list_cell_ends, n);
    gpu_set_all_zero(d_list_cell_included, n);
    gpu_set_all_zero(d_list_cell_idxs, n);
    gpu_set_all_zero(d_sum_cos, n * d);
    gpu_set_all_zero(d_sum_sin, n * d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_write_list_cell_id << < number_of_blocks, min(BLOCK_SIZE, n) >> >
                                                     (d_list_cell_dim_ids, d_D_current, n, d, width,
                                                             cell_size);

    //    printf("d_list_cell_dim_ids (%d*%d):\n", n, d);
    //    print_array_gpu((int *) d_list_cell_dim_ids, n, d);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_compute_list_cell_size << < number_of_blocks, min(BLOCK_SIZE, n) >> >
                                                         (d_list_cell_sizes, d_list_cell_included,
                                                                 d_list_cell_dim_ids,
                                                                 d_D_current, n, d, width, cell_size);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //    printf("d_list_cell_sizes (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_sizes, n);
    //
    //    printf("d_list_cell_included (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_included, n);

    inclusive_scan(d_list_cell_sizes, d_list_cell_ends, n);
    inclusive_scan(d_list_cell_included, d_list_cell_idxs, n);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //    printf("d_list_cell_ends (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_ends, n);
    //
    //    printf("d_list_cell_idxs (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_idxs, n);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_repack_list_cells << < number_of_blocks, min(BLOCK_SIZE, n) >> >
                                                    (d_new_list_cell_ends, d_new_list_cell_dim_ids,
                                                            d_list_cell_included, d_list_cell_idxs,
                                                            d_list_cell_ends,
                                                            d_list_cell_dim_ids,
                                                            n, d);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    swap(d_new_list_cell_ends, d_list_cell_ends);
    swap(d_new_list_cell_dim_ids, d_list_cell_dim_ids);

    //    printf("d_list_cell_ends (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_ends, n);
    //
    //    printf("d_list_cell_dim_ids (%d*%d):\n", n, d);
    //    print_array_gpu((int *) d_list_cell_dim_ids, n, d);

    gpu_set_all_zero(d_list_cell_sizes, n);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    kernel_populate_list_cell << < number_of_blocks, min(BLOCK_SIZE, n) >> > (d_sum_cos, d_sum_sin, d_list_cell_points,
            d_list_cell_ends,
            d_list_cell_sizes, d_list_cell_dim_ids,
            d_D_current,
            n, d, width, cell_size);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    //    printf("d_list_cell_points (%d):\n", n);
    //    print_array_gpu((int *) d_list_cell_points, n);
}

__global__ void
GPU_synCluster_list(int *__restrict__ d_C, const float *__restrict__ d_D_current,
                    unsigned int *__restrict__ d_list_cell_dim_ids, unsigned int *__restrict__ d_list_cell_ends,
                    const int n, const int d, const float cell_size, const int width) {

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int cell_idx = compute_cell_idx(d_D_current, d_list_cell_dim_ids, p, d, width, cell_size);

        int cell_start = get_start(d_list_cell_ends, cell_idx);
        int cell_end = get_end(d_list_cell_ends, cell_idx);
        int cell_number_of_points = cell_end - cell_start;

        if (cell_number_of_points > 1) {
            d_C[p] = cell_idx;
        }
    }
}

/// change termination criteria
clustering
GPU_DynamicalClustering_list(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d) / cell_factor;
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    //    int *d_list_cell_ids = gpu_malloc_int(n);
    unsigned int *d_list_cell_dim_ids = gpu_malloc_unsigned_int(n * d);
    unsigned int *d_new_list_cell_dim_ids = gpu_malloc_unsigned_int(n * d);
    unsigned int *d_list_cell_included = gpu_malloc_unsigned_int(n);
    unsigned int *d_list_cell_idxs = gpu_malloc_unsigned_int(n);
    unsigned int *d_list_cell_sizes = gpu_malloc_unsigned_int(n);
    unsigned int *d_list_cell_ends = gpu_malloc_unsigned_int(n);
    unsigned int *d_new_list_cell_ends = gpu_malloc_unsigned_int(n);
    unsigned int *d_list_cell_points = gpu_malloc_unsigned_int(n);
    float *d_sum_cos = gpu_malloc_float(n * d);
    float *d_sum_sin = gpu_malloc_float(n * d);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all(d_r_local, 1, 1.);

        build_list_grid(d_D_current,
                        d_list_cell_dim_ids, d_new_list_cell_dim_ids,
                        d_list_cell_included, d_list_cell_idxs,
                        d_list_cell_sizes, d_list_cell_ends,
                        d_new_list_cell_ends, d_list_cell_points,
                        d_sum_cos, d_sum_sin,
                        n, d, width, cell_size);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        int grid_number_of_cells = copy_last_D_to_H(d_list_cell_idxs, n);

        if (version == 1) {

            kernel_itr_list << < number_of_blocks, min(BLOCK_SIZE, n) >> > (d_D_next, d_r_local,
                    d_list_cell_ends, d_list_cell_dim_ids,
                    d_list_cell_points,
                    d_sum_sin, d_sum_cos, d_D_current,
                    n, d, grid_number_of_cells, eps, width,
                    cell_size);
        } else if (version == 2) {

            kernel_itr_list_2 << < number_of_blocks, min(BLOCK_SIZE, n) >> > (d_D_next, d_r_local,
                    d_list_cell_ends, d_list_cell_dim_ids,
                    d_list_cell_points,
                    d_sum_sin, d_sum_cos, d_D_current,
                    n, d, grid_number_of_cells, eps, width,
                    cell_size);
        }

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);

    build_list_grid(d_D_current,
                    d_list_cell_dim_ids, d_new_list_cell_dim_ids,
                    d_list_cell_included, d_list_cell_idxs,
                    d_list_cell_sizes, d_list_cell_ends,
                    d_new_list_cell_ends, d_list_cell_points,
                    d_sum_cos, d_sum_sin,
                    n, d, width, cell_size);
    gpuErrchk(cudaPeekAtLastError());

    GPU_synCluster_list << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_D_current, d_list_cell_dim_ids,
            d_list_cell_ends,
            n, d, cell_size, width);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);
    cudaFree(d_list_cell_dim_ids);
    cudaFree(d_new_list_cell_dim_ids);
    cudaFree(d_list_cell_included);
    cudaFree(d_list_cell_idxs);
    cudaFree(d_list_cell_sizes);
    cudaFree(d_list_cell_ends);
    cudaFree(d_new_list_cell_ends);
    cudaFree(d_list_cell_points);

    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD3(float *__restrict__ d_sum_cos, float *__restrict__ d_sum_sin,
                      float *__restrict__ d_r_local,
                      const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                      const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n,
                      const int d,
                      const float eps, const int width, const int grid_dims, const float cell_size,
                      int number_of_cells) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0;
        }
        float r_c = 0;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        int size = 0;
        do {

            idxs[0]++;

            {
                int l = 1;
                while (l < d && idxs[l - 1] > radius) {
                    idxs[l - 1] = -radius;
                    idxs[l]++;
                    l++;
                }
            }

            if (idxs[d - 1] > radius) {
                continue;
            }

            bool included = false;

            float small_dist = 0.;
            for (int l = 0; l < d; l++) {
                float left_coor = (tmp[l] + idxs[l]) * cell_size;
                float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                left_coor -= d_D_current[p * d + l];
                right_coor -= d_D_current[p * d + l];
                left_coor *= left_coor;
                right_coor *= right_coor;
                if (left_coor < right_coor) {
                    small_dist += left_coor;
                } else {
                    small_dist += right_coor;
                }
            }
            small_dist = sqrt(small_dist);

            if (small_dist <= eps) {
                included = true;
            }

            bool within_bounds = true;
            for (int l = 0; l < d; l++) {
                if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            size = 0;
            if (included && within_bounds) {
                number_of_cells = 1;
                cell = 0;
                for (int l = 0; l < d; l++) {
                    cell += (tmp[l] + idxs[l]) * number_of_cells;
                    number_of_cells *= width;
                }

                if (cell >= number_of_cells) {
                    printf("cell: %d\n", cell);
                }
                start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                end = cell < number_of_cells ? d_grid_ends[cell] : n;
                size = end - start;

                for (int l = 0; l < d; l++) {
                    const float x_t = d_D_current[p * d + l];
                    const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                               (d_sum_cos[cell * d + l] * sin(x_t)));
                    sum[l] += to_be_added;
                }
                number_of_neighbors += size;
            }
        } while (idxs[d - 1] <= radius);

        float sum_delta_y_i = 0;
        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        //        r_c = number_of_neighbors - sum_delta_y_i;
        //        r_c /= number_of_neighbors;
        //
        //        atomicAdd(d_r_local, r_c);//todo must be initialized before the kernel call
        number_of_cells = 1;
        cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell && d_r_local[0] > 0.5) {
            //            printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                   number_of_points_in_cell);
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

/// Adding all included cells as if there were fully included.
clustering GPU_DynamicalClustering_GRID_STAD3(float *h_D, int n, int d, float eps, float lam) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    int *d_grid = gpu_malloc_int(n);
    float *d_sum_cos = gpu_malloc_float(number_of_cells * d);
    float *d_sum_sin = gpu_malloc_float(number_of_cells * d);

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all(d_r_local, 1, 1.);
        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_grid_ends, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_cos, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_sin, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_STAD << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                           (d_sum_cos, d_sum_sin, d_D_current, n, d, cell_size, width,
                                                   grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_itr_grid_STAD3 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                     //        kernel_itr_grid_STAD1 << < 1, 1 >> >
                                                     (d_sum_cos, d_sum_sin, d_r_local, d_grid, d_grid_ends, d_D_current, d_D_next, n, d, eps,
                                                             width, grid_dims, cell_size, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1);
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    gpu_set_all_zero(d_grid_ends, number_of_cells);

    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, cell_size, width,
            grid_dims);

    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);

    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d,
                                                        cell_size, width, grid_dims);

    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d,
                                                       cell_size, grid_dims, width);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);

    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD4(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
                      float *__restrict__ d_r_local,
                      const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                      const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n,
                      const int d,
                      const float eps, const int width, const int grid_dims, const float cell_size,
                      int number_of_cells, int itr) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        float r_c = 0.;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        ///only not fully included
        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {

                    {
                        idxs[0]++;
                        int l = 1;
                        while (l < d && idxs[l - 1] > radius) {
                            idxs[l - 1] = -radius;
                            idxs[l]++;
                            l++;
                        }
                    }

                    if (idxs[d - 1] > radius) {
                        continue;
                    }

                    bool included = false;
                    fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    bool within_bounds = true;
                    for (int l = 0; l < d; l++) {
                        if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                            within_bounds = false;
                        }
                    }

                    size = 0;
                    if (included && within_bounds && !fully_included) {
                        number_of_cells = 1;
                        cell = 0;
                        for (int l = 0; l < d; l++) {
                            cell += (tmp[l] + idxs[l]) * number_of_cells;
                            number_of_cells *= width;
                        }

                        start = get_start(d_grid_ends, cell);
                        end = get_end(d_grid_ends, cell);
                        size = end - start;
                    }

                } while (size == 0 && idxs[d - 1] <= radius);

                if (idxs[d - 1] > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        ///only fully included
        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        end = -1;   // cell < width ? d_grid_ends[cell] : n;
        idx = -1;   // start
        int size = 0;
        fully_included = false;

        do {

            idxs[0]++;

            {
                int l = 1;
                while (l < d && idxs[l - 1] > radius) {
                    idxs[l - 1] = -radius;
                    idxs[l]++;
                    l++;
                }
            }

            if (idxs[d - 1] > radius) {
                continue;
            }

            float large_dist = 0.;
            for (int l = 0; l < d; l++) {
                float left_coor = (tmp[l] + idxs[l]) * cell_size;
                float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                left_coor -= d_D_current[p * d + l];
                right_coor -= d_D_current[p * d + l];
                left_coor *= left_coor;
                right_coor *= right_coor;
                if (left_coor < right_coor) {
                    large_dist += right_coor;
                } else {
                    large_dist += left_coor;
                }
            }
            large_dist = sqrt(large_dist);

            if (large_dist <= eps) {
                fully_included = true;
            }

            bool within_bounds = true;
            for (int l = 0; l < d; l++) {
                if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            size = 0;
            if (fully_included && within_bounds) {
                number_of_cells = 1;
                cell = 0;
                for (int l = 0; l < d; l++) {
                    cell += (tmp[l] + idxs[l]) * number_of_cells;
                    number_of_cells *= width;
                }

                start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                end = cell < number_of_cells ? d_grid_ends[cell] : n;
                size = end - start;

                for (int l = 0; l < d; l++) {
                    const float x_t = d_D_current[p * d + l];
                    const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                               (d_sum_cos[cell * d + l] * sin(x_t)));
                    sum[l] += to_be_added;
                }
                number_of_neighbors += size;
            }
        } while (idxs[d - 1] <= radius);

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        number_of_cells = 1;
        cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell) {
            //            if (itr > 10) {
            //                printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                       number_of_points_in_cell);
            //            }
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

/// adding first all included && !fully_included and then all fully_included
clustering GPU_DynamicalClustering_GRID_STAD4(float *h_D, int n, int d, float eps, float lam, int cell_factor) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d) / cell_factor;
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    int *d_grid = gpu_malloc_int(n);
    gpuErrchk(cudaPeekAtLastError());
    float *d_sum_cos = gpu_malloc_float(number_of_cells * d);
    gpuErrchk(cudaPeekAtLastError());
    float *d_sum_sin = gpu_malloc_float(number_of_cells * d);
    gpuErrchk(cudaPeekAtLastError());

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all(d_r_local, 1, 1.);
        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_grid_ends, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_cos, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_sin, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_STAD << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                           (d_sum_cos, d_sum_sin, d_D_current, n, d, cell_size, width,
                                                   grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        //        if (itr > 0) {
        //            for (int i = 0; i < width; i++) {
        //                print_array_nonzero_gpu(&d_grid_sizes[i * width], width);
        //                printf("\n");
        //            }
        //        }

        kernel_itr_grid_STAD4 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                     //        kernel_itr_grid_STAD1 << < 1, 1 >> >
                                                     (d_sum_cos, d_sum_sin, d_r_local, d_grid, d_grid_ends, d_D_current, d_D_next, n, d, eps,
                                                             width, grid_dims, cell_size, number_of_cells, itr);

        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1);
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    gpu_set_all_zero(d_grid_ends, number_of_cells);

    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, cell_size, width,
            grid_dims);

    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);

    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d,
                                                        cell_size, width, grid_dims);

    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d,
                                                       cell_size, grid_dims, width);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);

    delete h_C;

    return C;
}

__global__ void
kernel_itr_grid_STAD4_1(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
                        float *__restrict__ d_r_local,
                        const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
                        const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n,
                        const int d,
                        const float eps, const int width, const int grid_dims, const float cell_size,
                        int number_of_cells, int itr) {

    float *sum = new float[d];
    int *tmp = new int[d];
    int *idxs = new int[d];

    for (int idx1 = threadIdx.x + blockIdx.x * blockDim.x; idx1 < n; idx1 += blockDim.x * gridDim.x) {

        int p = d_grid[idx1];

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        float r_c = 0.;
        int number_of_neighbors = 0;

        for (int l = 0; l < d; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / cell_size;
            if (tmp[l] == width)
                tmp[l]--;
        }

        int radius = ceil(eps / cell_size);

        ///only not fully included
        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        int cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        int start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        int end = -1;   // cell < width ? d_grid_ends[cell] : n;
        int idx = -1;   // start
        bool fully_included = false;
        while (true) {

            if (idx >= end) {
                int size = 0;
                do {

                    {
                        idxs[0]++;
                        int l = 1;
                        while (l < d && idxs[l - 1] > radius) {
                            idxs[l - 1] = -radius;
                            idxs[l]++;
                            l++;
                        }
                    }

                    if (idxs[d - 1] > radius) {
                        continue;
                    }

                    bool included = false;
                    fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = (tmp[l] + idxs[l]) * cell_size;
                        float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];
                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {
                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }

                    bool within_bounds = true;
                    for (int l = 0; l < d; l++) {
                        if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                            within_bounds = false;
                        }
                    }

                    size = 0;
                    if (included && within_bounds && !fully_included) {
                        number_of_cells = 1;
                        cell = 0;
                        for (int l = 0; l < d; l++) {
                            cell += (tmp[l] + idxs[l]) * number_of_cells;
                            number_of_cells *= width;
                        }

                        start = get_start(d_grid_ends, cell);
                        end = get_end(d_grid_ends, cell);
                        size = end - start;
                    }

                } while (size == 0 && idxs[d - 1] <= radius);

                if (idxs[d - 1] > radius) {
                    break;
                }
                idx = start;
            }

            int q = d_grid[idx];

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
            }

            idx++;
        }

        ///only fully included
        idxs[0] = -radius - 1;
        for (int l = 1; l < d; l++) {
            idxs[l] = -radius;
        }

        cell = -1;  //(tmp_0 + i) + (tmp_1 + j) * width;
        start = -1; // cell > 0 ? d_grid_ends[cell - 1] : 0;
        end = -1;   // cell < width ? d_grid_ends[cell] : n;
        idx = -1;   // start
        int size = 0;
        fully_included = false;

        do {

            idxs[0]++;

            {
                int l = 1;
                while (l < d && idxs[l - 1] > radius) {
                    idxs[l - 1] = -radius;
                    idxs[l]++;
                    l++;
                }
            }

            if (idxs[d - 1] > radius) {
                continue;
            }

            float large_dist = 0.;
            for (int l = 0; l < d; l++) {
                float left_coor = (tmp[l] + idxs[l]) * cell_size;
                float right_coor = (tmp[l] + idxs[l] + 1) * cell_size;
                left_coor -= d_D_current[p * d + l];
                right_coor -= d_D_current[p * d + l];
                left_coor *= left_coor;
                right_coor *= right_coor;
                if (left_coor < right_coor) {
                    large_dist += right_coor;
                } else {
                    large_dist += left_coor;
                }
            }
            large_dist = sqrt(large_dist);

            if (large_dist <= eps) {
                fully_included = true;
            }

            bool within_bounds = true;
            for (int l = 0; l < d; l++) {
                if (tmp[l] + idxs[l] < 0 || width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            size = 0;
            if (fully_included && within_bounds) {
                number_of_cells = 1;
                cell = 0;
                for (int l = 0; l < d; l++) {
                    cell += (tmp[l] + idxs[l]) * number_of_cells;
                    number_of_cells *= width;
                }

                start = cell > 0 ? d_grid_ends[cell - 1] : 0;
                end = cell < number_of_cells ? d_grid_ends[cell] : n;
                size = end - start;

                for (int l = 0; l < d; l++) {
                    const float x_t = d_D_current[p * d + l];
                    const float to_be_added = ((d_sum_sin[cell * d + l] * cos(x_t)) -
                                               (d_sum_cos[cell * d + l] * sin(x_t)));
                    sum[l] += to_be_added;
                }
                number_of_neighbors += size;
            }
        } while (idxs[d - 1] <= radius);

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        number_of_cells = 1;
        cell = 0;
        for (int l = 0; l < d; l++) {
            cell += tmp[l] * number_of_cells;
            number_of_cells *= width;
        }

        int number_of_points_in_cell = get_end(d_grid_ends, cell) - get_start(d_grid_ends, cell);
        if (number_of_neighbors != number_of_points_in_cell) {
            //            if (itr > 10) {
            //                printf("number_of_neighbors: %d, number_of_points_in_cell: %d\n", number_of_neighbors,
            //                       number_of_points_in_cell);
            //            }
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

/// adding first all included && !fully_included and then all fully_included
clustering
GPU_DynamicalClustering_GRID_STAD4_1(float *h_D, int n, int d, float eps, float lam, int cell_factor) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    float cell_size = sqrt(pow((eps / 2.), 2.) / d) / cell_factor;
    int grid_dims = d;
    int number_of_cells = 1;
    int width = ceil(1. / cell_size);
    for (int i = 0; i < grid_dims; i++) {
        number_of_cells *= width;
    }

    int *d_grid_sizes = gpu_malloc_int(number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    int *d_grid_ends = gpu_malloc_int(number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    int *d_grid = gpu_malloc_int(n);
    gpuErrchk(cudaPeekAtLastError());
    float *d_sum_cos = gpu_malloc_float(number_of_cells * d);
    gpuErrchk(cudaPeekAtLastError());
    float *d_sum_sin = gpu_malloc_float(number_of_cells * d);
    gpuErrchk(cudaPeekAtLastError());

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;
    while (r_local < lam && itr < 100) {
        itr++;
        gpu_set_all(d_r_local, 1, 1.);
        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_grid_ends, number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_cos, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());
        gpu_set_all_zero(d_sum_sin, number_of_cells * d);
        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                 (d_grid_sizes, d_D_current, n, d, cell_size, width,
                                                         grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        gpu_set_all_zero(d_grid_sizes, number_of_cells);

        gpuErrchk(cudaPeekAtLastError());

        kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                    (d_grid, d_grid_ends, d_grid_sizes, d_D_current,
                                                            n, d, cell_size, width, grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        kernel_STAD << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                           (d_sum_cos, d_sum_sin, d_D_current, n, d, cell_size, width,
                                                   grid_dims);

        gpuErrchk(cudaPeekAtLastError());

        //        if (itr > 0) {
        //            for (int i = 0; i < width; i++) {
        //                print_array_nonzero_gpu(&d_grid_sizes[i * width], width);
        //                printf("\n");
        //            }
        //        }

        kernel_itr_grid_STAD4_1 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                       //        kernel_itr_grid_STAD1 << < 1, 1 >> >
                                                       (d_sum_cos, d_sum_sin, d_r_local, d_grid, d_grid_ends, d_D_current, d_D_next, n, d, eps,
                                                               width, grid_dims, cell_size, number_of_cells, itr);

        gpuErrchk(cudaPeekAtLastError());

        float *d_tmp = d_D_next;
        d_D_next = d_D_current;
        d_D_current = d_tmp;

        r_local = copy_last_D_to_H(d_r_local, 1);
        gpuErrchk(cudaPeekAtLastError());
        printf("r_local: %f\n", r_local);
    }

    clustering C;

    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);
    gpu_set_all_zero(d_grid_ends, number_of_cells);

    kernel_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_grid_sizes, d_D_current, n, d, cell_size, width,
            grid_dims);

    inclusive_scan(d_grid_sizes, d_grid_ends, number_of_cells);

    gpu_set_all_zero(d_grid_sizes, number_of_cells);

    kernel_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                (d_grid, d_grid_ends, d_grid_sizes, d_D_current, n, d,
                                                        cell_size, width, grid_dims);

    GPU_synCluster_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                               (d_C, d_incl, d_D_current, d_grid_ends, d_grid, n, d,
                                                       cell_size, grid_dims, width);

    inclusive_scan(d_incl, d_map, n);

    rename_cluster << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_map, n);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    cudaFree(d_D_current);
    cudaFree(d_D_next);
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);
    cudaFree(d_grid_sizes);
    cudaFree(d_grid_ends);
    cudaFree(d_grid);
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);
    cudaFree(d_r_local);

    delete h_C;

    return C;
}

__global__ void kernel_outer_grid_sizes(int *d_outer_grid_sizes, float *d_D_current, int n, int d,
                                        int outer_grid_width, int outer_grid_dims, float outer_cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);

        atomicInc((unsigned int *) &d_outer_grid_sizes[outer_cell], n);
    }
}

__global__ void
kernel_inner_grid_marking(int *d_inner_grid_cell_dim_ids, int *d_outer_grid_sizes, int *d_outer_grid_ends,
                          float *d_D_current,
                          int n, int d,
                          int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                          int inner_grid_width, int inner_grid_dims, float inner_cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
        int outer_cell_location = atomicInc((unsigned int *) &d_outer_grid_sizes[outer_cell], n);

        int inner_cell_idx = outer_cell_start + outer_cell_location;

        for (int i = 0; i < d; i++) {
            float val = d_D_current[p * d + i];
            int tmp = val / inner_cell_size;
            if (tmp == inner_grid_width)
                tmp--;
            d_inner_grid_cell_dim_ids[inner_cell_idx * d + i] = tmp;
        }
    }
}

__device__ bool
compute_inner_cell_matches(const float *d_D, const int *d_inner_grid_cell_dim_ids, const int inner_cell_idx,
                           const int p, const int d, const int inner_grid_width, const float inner_cell_size) {
    for (int i = 0; i < d; i++) {
        float val = d_D[p * d + i];
        int tmp = val / inner_cell_size;
        if (tmp == inner_grid_width)
            tmp--;
        if (d_inner_grid_cell_dim_ids[inner_cell_idx * d + i] != tmp) {
            return false;
        }
    }
    return true;
}

__device__ int compute_inner_cell_idx(const float *d_D_current, const int *d_inner_grid_cell_dim_ids,
                                      const int *d_outer_grid_ends, const int outer_cell, const int p, const int d,
                                      const int inner_grid_width, const float inner_cell_size) {

    int inner_cell_idx = get_start(d_outer_grid_ends, outer_cell);
    int end = get_end(d_outer_grid_ends, outer_cell);

    while (!compute_inner_cell_matches(d_D_current, d_inner_grid_cell_dim_ids, inner_cell_idx, p, d,
                                       inner_grid_width, inner_cell_size)) {
        inner_cell_idx++;
        if (inner_cell_idx >= end) {
            printf("it got to far!\n");
        }
    }

    return inner_cell_idx;
}


__device__ int compute_inner_cell_idx_test(const float *d_D_current, const int *d_inner_grid_cell_dim_ids,
                                           const int *d_outer_grid_ends, const int outer_cell, const int p, const int d,
                                           const int inner_grid_width, const float inner_cell_size, int n) {

    int inner_cell_idx = get_start(d_outer_grid_ends, outer_cell);

    while (!compute_inner_cell_matches(d_D_current, d_inner_grid_cell_dim_ids, inner_cell_idx, p, d,
                                       inner_grid_width, inner_cell_size)) {
        inner_cell_idx++;
        if (inner_cell_idx >= n) {
            printf("compute_inner_cell_idx_test:inner_cell_idx: %d\n", inner_cell_idx);
            return -1;
        }
    }

    return inner_cell_idx;
}

__global__ void
kernel_inner_grid_sizes(int *d_inner_grid_sizes, int *d_inner_grid_included, int *d_inner_grid_cell_dim_ids,
                        int *d_outer_grid_ends,
                        float *d_D_current, int n, int d,
                        int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                        int inner_grid_width, int inner_grid_dims, float inner_cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);

        int inner_cell_idx = compute_inner_cell_idx_test(d_D_current, d_inner_grid_cell_dim_ids, d_outer_grid_ends,
                                                         outer_cell, p, d,
                                                         inner_grid_width, inner_cell_size, n);
        if (inner_cell_idx >= n) {
            printf("inner_cell_idx: %d\n", inner_cell_idx);
        }

        atomicInc((unsigned int *) &d_inner_grid_sizes[inner_cell_idx], n);

        d_inner_grid_included[inner_cell_idx] = 1;
    }
}

__global__ void
kernel_inner_grid_populate(int *d_inner_grid_points, int *d_inner_grid_sizes, int *d_inner_grid_cell_dim_ids,
                           int *d_inner_grid_ends,
                           int *d_outer_grid_ends,
                           float *d_D_current,
                           int n, int d,
                           int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                           int inner_grid_width, int inner_grid_dims, float inner_cell_size) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);

        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_grid_cell_dim_ids, d_outer_grid_ends,
                                                    outer_cell, p, d,
                                                    inner_grid_width, inner_cell_size);

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);

        int point_location = atomicInc((unsigned int *) &d_inner_grid_sizes[inner_cell_idx], n);
        int point_idx = inner_cell_start + point_location;

        d_inner_grid_points[point_idx] = p;
    }
}

__global__ void kernel_outer_grid_repack(int *d_new_outer_grid_ends, int *d_outer_grid_ends, int *d_inner_grid_idxs,
                                         int outer_grid_number_of_cells) {
    for (int outer_cell = threadIdx.x + blockIdx.x * blockDim.x;
         outer_cell < outer_grid_number_of_cells;
         outer_cell += blockDim.x * gridDim.x) {

        int outer_grid_end = get_end(d_outer_grid_ends, outer_cell);

        int inner_grid_idx = outer_grid_end > 0 ? d_inner_grid_idxs[outer_grid_end - 1] : 0;

        d_new_outer_grid_ends[outer_cell] = inner_grid_idx;
    }
}

__global__ void
kernel_inner_grid_repack(int *d_new_inner_grid_ends, int *d_new_inner_grid_cell_dim_ids, int *d_inner_grid_idxs,
                         int *d_inner_grid_included, int *d_inner_grid_ends, int *d_inner_cell_dim_ids,
                         int n, int d) {
    for (int old_inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         old_inner_cell_idx < n; old_inner_cell_idx += blockDim.x * gridDim.x) {
        if (d_inner_grid_included[old_inner_cell_idx] > 0) {
            int new_inner_cell_idx = d_inner_grid_idxs[old_inner_cell_idx] - 1;
            int new_inner_grid_end = get_end(d_inner_grid_ends, old_inner_cell_idx);

            d_new_inner_grid_ends[new_inner_cell_idx] = new_inner_grid_end;

            for (int i = 0; i < d; i++) {
                d_new_inner_grid_cell_dim_ids[new_inner_cell_idx * d + i] = d_inner_cell_dim_ids[
                        old_inner_cell_idx * d + i];
            }
        }
    }
}

__global__ void
kernel_inner_grid_stats(float *d_sum_cos, float *d_sum_sin,
                        int *d_outer_grid_ends,
                        int *d_inner_grid_ends, int *d_inner_cell_dim_ids,
                        float *d_D_current,
                        int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                        int inner_grid_width, float inner_cell_size,
                        int n, int d) {
    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);

        for (int i = 0; i < d; i++) {
            float val = d_D_current[p * d + i];
            atomicAdd(&d_sum_cos[inner_cell_idx * d + i], cos(val));
            atomicAdd(&d_sum_sin[inner_cell_idx * d + i], sin(val));
        }
    }
}

__global__ void
kernel_itr_grid_STAD5(float *__restrict__ d_r_local,
                      const int *__restrict__ d_outer_grid_ends,
                      const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                      const int *__restrict__ d_inner_cell_dim_ids,
                      float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                      const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                      const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                      const int inner_grid_width, const float inner_cell_size,
                      const int n, const int d, const float eps, int itr) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    extern __shared__ float s_d[];

//    float *sum = new float[d];
//    int *tmp = new int[outer_grid_dims];
//    int *idxs = new int[outer_grid_dims];

    float *sum = &s_d[threadIdx.x * (d + 2 * outer_grid_dims)];
    int *tmp = (int *) &sum[d];
    int *idxs = &tmp[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        int p = d_inner_cell_points[p_idx];

        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        bool on_boarder = false;
        for (int l = 0; l < d; l++) {
            sum[l] = 0.;


            int idm = (int) ((d_D_current[p * d + l] - 0.00001) / inner_cell_size);
            int id = (int) (d_D_current[p * d + l] / inner_cell_size);
            int idp = (int) ((d_D_current[p * d + l] + 0.00001) / inner_cell_size);


            if (id != idm || id != idp)
                on_boarder = true;
        }
        float r_c = 0.;
        int number_of_neighbors = 0;
        int half_count = 0;

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

//                if (p == 21965) {
//                    if (outer_cell_number_of_cells > 0) {
//                        printf("outer_cell: %d\n", outer_cell);
//                    }
//                }

                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps) {

                            included = true;
                        }

                        if (large_dist <= eps) {
                            fully_included = true;
                        }

                        if (fully_included) {
                            for (int l = 0; l < d; l++) {
                                const float x_t = d_D_current[p * d + l];
                                const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                                sum[l] += to_be_added;

//                                if (p == 21965) {
//                                    printf("inner_cell_idx: %d, d_sum_sin[inner_cell_idx * d + l]: %f, d_sum_cos[inner_cell_idx * d + l]: %f\n",
//                                           inner_cell_idx, d_sum_sin[inner_cell_idx * d + l],
//                                           d_sum_cos[inner_cell_idx * d + l]);
//                                }
                            }
                            number_of_neighbors += inner_cell_number_of_points;

                            if (on_boarder) {
                                if (tmp_inner_cell_idx == inner_cell_idx) {
                                    half_count += inner_cell_number_of_points;
                                } else {
                                    for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                        int q = d_inner_cell_points[q_idx];
                                        float dist = gpu_distance(p, q, d_D_current, d);
                                        if (dist <= eps / 2.) {
                                            for (int l = 0; l < d; l++) {
                                                sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                                            }
                                            half_count++;
                                        }
                                    }
                                }
                            }
                        } else if (included) {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];

                                if (gpu_distance(p, q, d_D_current, d) <= eps) {
                                    for (int l = 0; l < d; l++) {
                                        sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                                    }
                                    number_of_neighbors++;
                                }
                            }
                        }
                    }
                }
            }
            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        if ((!on_boarder && number_of_neighbors != inner_cell_number_of_points) ||
            (on_boarder && number_of_neighbors != half_count)) {
            d_r_local[0] = 0.;
            if (itr > 30 && 22000 > inner_cell_number_of_points && 20000 < inner_cell_number_of_points && p== 87378 ) {
                printf("itr: %d, p: %d, c: %d full: %d, half: %d, lower bound half: %d, on_boarder: %d\n", itr, p,
                       tmp_inner_cell_idx,
                       number_of_neighbors, half_count,
                       inner_cell_number_of_points, on_boarder ? 1 : 0);

                printf("update: ");
                for (int l = 0; l < d; l++) {
                    printf("%0.10f ", sum[l] / number_of_neighbors);
                }
                printf("\n");
            }
        }
    }
//    delete sum;
//    delete tmp;
//    delete idxs;
}

__global__
void kernel_itr_grid_STAD5_5non_empty(int *d_pre_grid_non_empty, int *d_pre_grid_sizes, int *d_outer_grid_ends,
                                      int outer_grid_number_of_cells) {
    for (int outer_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         outer_cell_idx < outer_grid_number_of_cells;
         outer_cell_idx += blockDim.x * gridDim.x) {

        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

        if (outer_cell_number_of_cells != 0) {
            int loc = atomicInc((unsigned int *) &d_pre_grid_sizes[0], outer_grid_number_of_cells);
            d_pre_grid_non_empty[loc] = outer_cell_idx;
        }
    }
}

__global__
void kernel_itr_grid_STAD5_6non_empty(int *d_pre_grid_non_empty, int *d_pre_grid_sizes, int *d_inner_grid_ends,
                                      int inner_grid_number_of_cells) {
    for (int inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         inner_cell_idx < inner_grid_number_of_cells;
         inner_cell_idx += blockDim.x * gridDim.x) {

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        if (inner_cell_number_of_points != 0) {
            int loc = atomicInc((unsigned int *) &d_pre_grid_sizes[0], inner_grid_number_of_cells);
            d_pre_grid_non_empty[loc] = inner_cell_idx;
        }
    }
}

__global__
void kernel_itr_grid_STAD5_5pre_size(int *d_pre_grid_sizes, int *d_outer_grid_ends,
                                     int outer_grid_number_of_cells, int outer_grid_dims, int outer_grid_width,
                                     float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    int *tmp = new int[outer_grid_dims];
    int *idxs = new int[outer_grid_dims];

    for (int outer_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         outer_cell_idx < outer_grid_number_of_cells;
         outer_cell_idx += blockDim.x * gridDim.x) {

        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

        if (outer_cell_number_of_cells == 0) {
            continue;
        }

        int value = outer_cell_idx;
        for (int l = 0; l < outer_grid_dims; l++) {
            tmp[l] = value % outer_grid_width;
            value /= outer_grid_width;
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                /// check size
                int other_outer_number_of_cells = 1;
                int other_outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    other_outer_cell += (tmp[l] + idxs[l]) * other_outer_number_of_cells;
                    other_outer_number_of_cells *= outer_grid_width;
                }

                int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_number_of_cells = other_outer_cell_end - other_outer_cell_start;
                if (other_outer_cell_number_of_cells > 0) {
                    atomicInc((unsigned int *) &d_pre_grid_sizes[outer_cell_idx], outer_grid_number_of_cells);
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }

    delete tmp;
    delete idxs;
}


__global__
void kernel_itr_grid_STAD5_5pre_size_1(int *d_pre_grid_non_empty, int *d_pre_grid_sizes, int *d_outer_grid_ends,
                                       int non_empty_cells, int outer_grid_dims, int outer_grid_width,
                                       float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    //todo use shared arrays
    int *tmp = new int[outer_grid_dims];
    int *idxs = new int[outer_grid_dims];

    for (int loc = threadIdx.x + blockIdx.x * blockDim.x;
         loc < non_empty_cells;
         loc += blockDim.x * gridDim.x) {
        int outer_cell_idx = d_pre_grid_non_empty[loc];

        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

        int value = outer_cell_idx;
        for (int l = 0; l < outer_grid_dims; l++) {
            tmp[l] = value % outer_grid_width;
            value /= outer_grid_width;
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                /// check size
                int other_outer_number_of_cells = 1;
                int other_outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    other_outer_cell += (tmp[l] + idxs[l]) * other_outer_number_of_cells;
                    other_outer_number_of_cells *= outer_grid_width;
                }

                int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_number_of_cells = other_outer_cell_end - other_outer_cell_start;
                if (other_outer_cell_number_of_cells > 0) {
                    atomicInc((unsigned int *) &d_pre_grid_sizes[outer_cell_idx], non_empty_cells);
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }

    delete tmp;
    delete idxs;
}

__global__
void kernel_itr_grid_STAD5_6pre_size(int *d_pre_grid_ends, int *d_pre_grid_cells,
                                     int *d_pre_inner_grid_sizes, int *d_pre_inner_grid_fully_sizes,
                                     int *d_outer_grid_ends,
                                     int *d_inner_grid_ends, int *d_inner_grid_cell_dim_ids,
                                     int non_empty_cells, int outer_grid_dims, int outer_grid_width, int d,
                                     float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    for (int inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         inner_cell_idx < non_empty_cells;
         inner_cell_idx += blockDim.x * gridDim.x) {

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        if (inner_cell_number_of_points == 0) {
            continue;
        }

        int number_of_cells = 1;
        int outer_cell = 0;
        for (int l = 0; l < outer_grid_dims; l++) {
            int tmp = d_inner_grid_cell_dim_ids[inner_cell_idx * d + l];
            outer_cell += tmp * number_of_cells;
            number_of_cells *= outer_grid_width;
        }

        int pre_start = get_start(d_pre_grid_ends, outer_cell);
        int pre_end = get_end(d_pre_grid_ends, outer_cell);

        for (int loc = pre_start; loc < pre_end; loc++) {
            int other_outer_cell = d_pre_grid_cells[loc];
            int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
            int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
            for (int other_inner_cell_idx = other_outer_cell_start;
                 other_inner_cell_idx < other_outer_cell_end; other_inner_cell_idx++) {

                int all_close = 0;
                for (int l = 0; l < d; l++) {
                    int dist = abs(d_inner_grid_cell_dim_ids[other_inner_cell_idx * d + l] -
                                   d_inner_grid_cell_dim_ids[inner_cell_idx * d + l]);
                    if (dist > all_close) {
                        all_close = dist;
                    }
                }
                if (all_close <= 1) {
                    atomicInc((unsigned int *) &d_pre_inner_grid_fully_sizes[inner_cell_idx], non_empty_cells);
                } else if (all_close <= 3) {
                    atomicInc((unsigned int *) &d_pre_inner_grid_sizes[inner_cell_idx], non_empty_cells);
                }
            }
        }
    }
}

__global__
void kernel_itr_grid_STAD5_6_1pre_size(int *d_pre_grid_ends, int *d_pre_grid_cells,
                                       int *d_pre_inner_grid_sizes, int *d_pre_inner_grid_fully_sizes,
                                       int *d_outer_grid_ends,
                                       int *d_inner_grid_ends, int *d_inner_grid_cell_dim_ids,
                                       int non_empty_cells, int outer_grid_dims, int outer_grid_width, int d,
                                       float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    for (int inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         inner_cell_idx < non_empty_cells;
         inner_cell_idx += blockDim.x * gridDim.x) {

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        if (inner_cell_number_of_points == 0) {
            continue;
        }

        int number_of_cells = 1;
        int outer_cell = 0;
        for (int l = 0; l < outer_grid_dims; l++) {
            int tmp = d_inner_grid_cell_dim_ids[inner_cell_idx * d + l];
            outer_cell += tmp * number_of_cells;
            number_of_cells *= outer_grid_width;
        }

        int pre_start = get_start(d_pre_grid_ends, outer_cell);
        int pre_end = get_end(d_pre_grid_ends, outer_cell);

        for (int loc = pre_start; loc < pre_end; loc++) {
            int other_outer_cell = d_pre_grid_cells[loc];
            int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
            int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
            for (int other_inner_cell_idx = other_outer_cell_start;
                 other_inner_cell_idx < other_outer_cell_end; other_inner_cell_idx++) {

                int all_close = 0;
                for (int l = 0; l < d; l++) {
                    int dist = abs(d_inner_grid_cell_dim_ids[other_inner_cell_idx * d + l] -
                                   d_inner_grid_cell_dim_ids[inner_cell_idx * d + l]);
                    if (dist > all_close) {
                        all_close = dist;
                    }
                }
                if (all_close <= 1) {
                    atomicInc((unsigned int *) &d_pre_inner_grid_fully_sizes[inner_cell_idx], non_empty_cells);
                } else if (all_close <= 3) {
                    const int other_inner_cell_start = get_start(d_inner_grid_ends, other_inner_cell_idx);
                    const int other_inner_cell_end = get_end(d_inner_grid_ends, other_inner_cell_idx);
                    const int other_inner_cell_size = other_inner_cell_end - other_inner_cell_start;
                    atomicAdd((unsigned int *) &d_pre_inner_grid_sizes[inner_cell_idx], other_inner_cell_size);
                }
            }
        }
    }
}

__global__
void kernel_itr_grid_STAD5_5pre_populate(int *d_pre_grid_cells, int *d_pre_grid_ends, int *d_pre_grid_sizes,
                                         int *d_outer_grid_ends,
                                         int outer_grid_number_of_cells, int outer_grid_dims, int outer_grid_width,
                                         float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    int *tmp = new int[outer_grid_dims];
    int *idxs = new int[outer_grid_dims];

    for (int outer_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         outer_cell_idx < outer_grid_number_of_cells;
         outer_cell_idx += blockDim.x * gridDim.x) {

        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

        if (outer_cell_number_of_cells == 0) {
            continue;
        }

        int value = outer_cell_idx;
        for (int l = 0; l < outer_grid_dims; l++) {
            tmp[l] = value % outer_grid_width;
            value /= outer_grid_width;
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                /// check size
                int other_outer_number_of_cells = 1;
                int other_outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    other_outer_cell += (tmp[l] + idxs[l]) * other_outer_number_of_cells;
                    other_outer_number_of_cells *= outer_grid_width;
                }

                int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_number_of_cells = other_outer_cell_end - other_outer_cell_start;
                if (other_outer_cell_number_of_cells > 0) {
                    int offset = get_start(d_pre_grid_ends, outer_cell_idx);
                    int loc = atomicInc((unsigned int *) &d_pre_grid_sizes[outer_cell_idx], outer_grid_number_of_cells);
                    d_pre_grid_cells[offset + loc] = other_outer_cell;
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }

    delete tmp;
    delete idxs;
}

__global__
void kernel_itr_grid_STAD5_5pre_populate_1(int *d_pre_grid_non_empty, int *d_pre_grid_cells, int *d_pre_grid_ends,
                                           int *d_pre_grid_sizes,
                                           int *d_outer_grid_ends,
                                           int non_empty_cells, int outer_grid_dims, int outer_grid_width,
                                           float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    int *tmp = new int[outer_grid_dims];//pre allocated these
    int *idxs = new int[outer_grid_dims];

    for (int loc = threadIdx.x + blockIdx.x * blockDim.x;
         loc < non_empty_cells;
         loc += blockDim.x * gridDim.x) {
        int outer_cell_idx = d_pre_grid_non_empty[loc];

        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell_idx);
        int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;

        if (outer_cell_number_of_cells == 0) {
            continue;
        }

        int value = outer_cell_idx;
        for (int l = 0; l < outer_grid_dims; l++) {
            tmp[l] = value % outer_grid_width;
            value /= outer_grid_width;
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                /// check size
                int other_outer_number_of_cells = 1;
                int other_outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    other_outer_cell += (tmp[l] + idxs[l]) * other_outer_number_of_cells;
                    other_outer_number_of_cells *= outer_grid_width;
                }

                int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
                int other_outer_cell_number_of_cells = other_outer_cell_end - other_outer_cell_start;
                if (other_outer_cell_number_of_cells > 0) {
                    int offset = get_start(d_pre_grid_ends, outer_cell_idx);
                    int loc = atomicInc((unsigned int *) &d_pre_grid_sizes[outer_cell_idx], non_empty_cells);

                    d_pre_grid_cells[offset + loc] = other_outer_cell;
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }

    delete tmp;
    delete idxs;
}

__global__
void kernel_itr_grid_STAD5_6pre_populate(int *d_pre_grid_ends, int *d_pre_grid_cells,
                                         int *d_pre_inner_grid_cells, int *d_pre_inner_grid_ends,
                                         int *d_pre_inner_grid_sizes,
                                         int *d_pre_inner_grid_fully_cells, int *d_pre_inner_grid_fully_ends,
                                         int *d_pre_inner_grid_fully_sizes,
                                         int *d_outer_grid_ends,
                                         int *d_inner_grid_ends, int *d_inner_grid_cell_dim_ids,
                                         int non_empty_cells, int outer_grid_dims, int outer_grid_width, int d,
                                         float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    for (int inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         inner_cell_idx < non_empty_cells;
         inner_cell_idx += blockDim.x * gridDim.x) {

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        if (inner_cell_number_of_points == 0) {
            continue;
        }

        int number_of_cells = 1;
        int outer_cell = 0;
        for (int l = 0; l < outer_grid_dims; l++) {
            int tmp = d_inner_grid_cell_dim_ids[inner_cell_idx * d + l];
            outer_cell += tmp * number_of_cells;
            number_of_cells *= outer_grid_width;
        }

        int pre_start = get_start(d_pre_grid_ends, outer_cell);
        int pre_end = get_end(d_pre_grid_ends, outer_cell);

        for (int loc = pre_start; loc < pre_end; loc++) {
            int other_outer_cell = d_pre_grid_cells[loc];

            int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
            int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
            for (int other_inner_cell_idx = other_outer_cell_start;
                 other_inner_cell_idx < other_outer_cell_end; other_inner_cell_idx++) {

                int all_close = 0;
                for (int l = 0; l < d; l++) {
                    int dist = abs(d_inner_grid_cell_dim_ids[other_inner_cell_idx * d + l] -
                                   d_inner_grid_cell_dim_ids[inner_cell_idx * d + l]);
                    if (dist > all_close) {
                        all_close = dist;
                    }
                }
                if (all_close <= 1) {
                    int offset = get_start(d_pre_inner_grid_fully_ends, inner_cell_idx);
                    int loc2 = atomicInc((unsigned int *) &d_pre_inner_grid_fully_sizes[inner_cell_idx],
                                         non_empty_cells);
                    d_pre_inner_grid_fully_cells[offset + loc2] = other_inner_cell_idx;
                } else if (all_close <= 3) {
                    int offset = get_start(d_pre_inner_grid_ends, inner_cell_idx);
                    int loc2 = atomicInc((unsigned int *) &d_pre_inner_grid_sizes[inner_cell_idx], non_empty_cells);
                    d_pre_inner_grid_cells[offset + loc2] = other_inner_cell_idx;
                }
            }
        }
    }
}

__global__
void kernel_itr_grid_STAD5_6_1pre_populate(int *d_pre_grid_ends, int *d_pre_grid_cells,
                                           int *d_pre_inner_grid_cells, int *d_pre_inner_grid_ends,
                                           int *d_pre_inner_grid_sizes,
                                           int *d_pre_inner_grid_fully_cells, int *d_pre_inner_grid_fully_ends,
                                           int *d_pre_inner_grid_fully_sizes,
                                           int *d_outer_grid_ends,
                                           int *d_inner_grid_points, int *d_inner_grid_ends,
                                           int *d_inner_grid_cell_dim_ids,
                                           int non_empty_cells, int outer_grid_dims, int outer_grid_width, int d,
                                           float eps, float outer_cell_size) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    for (int inner_cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
         inner_cell_idx < non_empty_cells;
         inner_cell_idx += blockDim.x * gridDim.x) {

        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        if (inner_cell_number_of_points == 0) {
            continue;
        }

        int number_of_cells = 1;
        int outer_cell = 0;
        for (int l = 0; l < outer_grid_dims; l++) {
            int tmp = d_inner_grid_cell_dim_ids[inner_cell_idx * d + l];
            outer_cell += tmp * number_of_cells;
            number_of_cells *= outer_grid_width;
        }

        int pre_start = get_start(d_pre_grid_ends, outer_cell);
        int pre_end = get_end(d_pre_grid_ends, outer_cell);

        for (int loc = pre_start; loc < pre_end; loc++) {//todo check loc?
            int other_outer_cell = d_pre_grid_cells[loc];

            int other_outer_cell_start = get_start(d_outer_grid_ends, other_outer_cell);
            int other_outer_cell_end = get_end(d_outer_grid_ends, other_outer_cell);
            for (int other_inner_cell_idx = other_outer_cell_start;
                 other_inner_cell_idx < other_outer_cell_end; other_inner_cell_idx++) {

                int all_close = 0;
                for (int l = 0; l < d; l++) {
                    int dist = abs(d_inner_grid_cell_dim_ids[other_inner_cell_idx * d + l] -
                                   d_inner_grid_cell_dim_ids[inner_cell_idx * d + l]);
                    if (dist > all_close) {
                        all_close = dist;
                    }
                }
                if (all_close <= 1) {
                    int offset = get_start(d_pre_inner_grid_fully_ends, inner_cell_idx);
                    int loc2 = atomicInc((unsigned int *) &d_pre_inner_grid_fully_sizes[inner_cell_idx],
                                         non_empty_cells);
                    d_pre_inner_grid_fully_cells[offset + loc2] = other_inner_cell_idx;
                } else if (all_close <= 3) {

                    const int other_inner_cell_start = get_start(d_inner_grid_ends, other_inner_cell_idx);
                    const int other_inner_cell_end = get_end(d_inner_grid_ends, other_inner_cell_idx);

                    int offset = get_start(d_pre_inner_grid_ends, inner_cell_idx);
                    for (int p_idx = other_inner_cell_start; p_idx < other_inner_cell_end; p_idx++) {
                        int p = d_inner_grid_points[p_idx];
                        int loc2 = atomicInc((unsigned int *) &d_pre_inner_grid_sizes[inner_cell_idx], non_empty_cells);
//                        if(offset + loc2>=non_empty_cells){
//                            printf("offset: %d + loc2: %d = %d >= non_empty_cells: %d\n", offset, loc2, offset + loc2, non_empty_cells);
//                        }
                        d_pre_inner_grid_cells[offset + loc2] = p;
                    }
                }
            }
        }
    }
}

__global__ void
kernel_itr_grid_STAD5_5(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                        float *__restrict__ d_r_local,
                        const int *__restrict__ d_outer_grid_ends,
                        const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                        const int *__restrict__ d_inner_cell_dim_ids,
                        float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                        const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                        const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                        const int inner_grid_width, const float inner_cell_size,
                        const int n, const int d, const float eps) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = new float[d];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        const int p = d_inner_cell_points[p_idx];

        const int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                   outer_cell_size);

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        int number_of_neighbors = 0;

        const int pre_start = get_start(d_pre_grid_ends, tmp_outer_cell);
        const int pre_end = get_end(d_pre_grid_ends, tmp_outer_cell);
        for (int loc = pre_start; loc < pre_end; loc++) {
            const int outer_cell = d_pre_grid_cells[loc];

            const int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
            const int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
            for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {

                ///compute included
                bool included = false;
                bool fully_included = false;

                float small_dist = 0.;
                float large_dist = 0.;
                for (int l = 0; l < d; l++) {
                    float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                    float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                    left_coor -= d_D_current[p * d + l];
                    right_coor -= d_D_current[p * d + l];

                    left_coor *= left_coor;
                    right_coor *= right_coor;
                    if (left_coor < right_coor) {
                        small_dist += left_coor;
                        large_dist += right_coor;
                    } else {
                        small_dist += right_coor;
                        large_dist += left_coor;
                    }
                }
                small_dist = sqrt(small_dist);
                large_dist = sqrt(large_dist);

                if (small_dist <= eps) {

                    included = true;
                }

                if (large_dist <= eps) {
                    fully_included = true;
                }

                const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;
                if (fully_included) {
                    for (int l = 0; l < d; l++) {
                        const float x_t = d_D_current[p * d + l];
                        const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                   (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                        sum[l] += to_be_added;
                    }
                    number_of_neighbors += inner_cell_number_of_points;
                }
                if (!fully_included && included) {
                    for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                        int q = d_inner_cell_points[q_idx];

                        if (gpu_distance(p, q, d_D_current, d) <= eps) {
                            for (int l = 0; l < d; l++) {
                                sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                            }
                            number_of_neighbors++;
                        }
                    }
                }
            }
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        if (number_of_neighbors != inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
    delete sum;
}

__global__ void
kernel_itr_grid_STAD5_5_1(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps, int itr) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    extern __shared__ float s_d[];

//    float *sum = new float[d];
    float *sum = &s_d[threadIdx.x * d * 2];
    float *x = &sum[d];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        const int p = d_inner_cell_points[p_idx];

        const int center_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                      outer_cell_size);
        int center_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                           center_outer_cell, p, d, inner_grid_width, inner_cell_size);
        int center_inner_cell_start = get_start(d_inner_grid_ends, center_inner_cell_idx);
        int center_inner_cell_end = get_end(d_inner_grid_ends, center_inner_cell_idx);
        int center_inner_cell_number_of_points = center_inner_cell_end - center_inner_cell_start;

        bool on_boarder = false;

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
            x[l] = d_D_current[p * d + l];

            int idm = (int) ((x[l] - 0.00001) / inner_cell_size);
            int id = (int) (x[l] / inner_cell_size);
            int idp = (int) ((x[l] + 0.00001) / inner_cell_size);


            if (id != idm || id != idp)
                on_boarder = true;
        }

        int number_of_neighbors = 0;
        int half_count = 0;

        const int pre_start = get_start(d_pre_grid_ends, center_outer_cell);
        const int pre_end = get_end(d_pre_grid_ends, center_outer_cell);

        for (int loc = pre_start; loc < pre_end; loc++) {
            const int outer_cell = d_pre_grid_cells[loc];

            const int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
            const int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
            for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {

                ///compute included
                float small_dist = 0.;
                float large_dist = 0.;
                for (int l = 0; l < d; l++) {
                    int dim_id = d_inner_cell_dim_ids[inner_cell_idx * d + l];
                    float left_coor = dim_id * inner_cell_size;
                    float right_coor = (dim_id + 1) * inner_cell_size;
                    left_coor -= x[l];
                    right_coor -= x[l];

                    left_coor *= left_coor;
                    right_coor *= right_coor;
                    if (left_coor < right_coor) {
                        small_dist += left_coor;
                        large_dist += right_coor;
                    } else {
                        small_dist += right_coor;
                        large_dist += left_coor;
                    }
                }
                small_dist = sqrt(small_dist);
                large_dist = sqrt(large_dist);

                bool included = small_dist <= eps;
                bool fully_included = large_dist <= eps;

                const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;
                if (fully_included) {
                    for (int l = 0; l < d; l++) {

//                        sum[l] += ((d_sum_sin[inner_cell_idx * d + l] * cos(x[l])) -
//                                   (d_sum_cos[inner_cell_idx * d + l] * sin(x[l])));

                        const float x_t = d_D_current[p * d + l];
                        const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                   (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                        sum[l] += to_be_added;

                    }
                    number_of_neighbors += inner_cell_number_of_points;


                    if (on_boarder) {
                        if (center_inner_cell_idx == inner_cell_idx) {
                            half_count += inner_cell_number_of_points;
                        } else {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];
                                float dist = gpu_distance(p, q, d_D_current, d);
                                if (dist <= eps / 2.) {
                                    for (int l = 0; l < d; l++) {
                                        sum[l] += sin(d_D_current[q * d + l] - x[l]);
                                    }
                                    half_count++;
                                }
                            }
                        }
                    }

                } else if (included) {
                    for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                        int q = d_inner_cell_points[q_idx];
                        float dist = gpu_distance(p, q, d_D_current, d);
                        if (dist <= eps) {
                            for (int l = 0; l < d; l++) {
                                sum[l] += sin(d_D_current[q * d + l] - x[l]);
                            }
                            number_of_neighbors++;
                        }
                    }
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = x[l] + sum[l] / number_of_neighbors;
        }


        if ((!on_boarder && number_of_neighbors != center_inner_cell_number_of_points) ||
            (on_boarder && number_of_neighbors != half_count)) {
            d_r_local[0] = 0.;
            if (itr > 30 && 10 > center_inner_cell_number_of_points) {

//                printf("itr: %d, p: %d, c: %d full: %d, half: %d, lower bound half: %d, on_boarder: %d\n", itr, p,
//                       center_inner_cell_idx,
//                       number_of_neighbors, half_count,
//                       center_inner_cell_number_of_points, on_boarder ? 1 : 0);
//
//                printf("update: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.7f ", sum[l] / number_of_neighbors);
//                }
//                printf("\n");
//
//                printf("x: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.7f ", x[l]);
//                }
//                printf("\n");
//                printf("id: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%d ", (int) (x[l] / outer_cell_size));
//                }
//                printf("\n");
            }
        }
    }
}


__global__ void
kernel_itr_grid_STAD5_5_3(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps) {

    const int group_size = 16;
    const int outer_grid_radius = ceil(eps / outer_cell_size);

    extern __shared__ float s_d[];

//    float *sum = new float[d];
    float *sum = &s_d[threadIdx.x * (d * 2 + group_size)];
    float *x = &sum[d];
    int *s_included = (int *) &x[d];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        const int p = d_inner_cell_points[p_idx];

        const int center_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                      outer_cell_size);
        int center_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                           center_outer_cell, p, d, inner_grid_width, inner_cell_size);
        int center_inner_cell_start = get_start(d_inner_grid_ends, center_inner_cell_idx);
        int center_inner_cell_end = get_end(d_inner_grid_ends, center_inner_cell_idx);
        int center_inner_cell_number_of_points = center_inner_cell_end - center_inner_cell_start;

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
            x[l] = d_D_current[p * d + l];
        }

        int number_of_neighbors = 0;

        const int pre_start = get_start(d_pre_grid_ends, center_outer_cell);
        const int pre_end = get_end(d_pre_grid_ends, center_outer_cell);

        int loc = pre_start;

        int outer_cell = d_pre_grid_cells[loc];
        int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
        int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);

        int inner_cell_idx = outer_cell_start;

        while (loc < pre_end) {
            int group_counter = 0;
            int included_size = 0;


            while (loc < pre_end && group_counter < group_size) {


//                while (inner_cell_idx < outer_cell_end && group_counter < group_size) {

                ///compute included
                float small_dist = 0.;
                float large_dist = 0.;
                for (int l = 0; l < d; l++) {
                    int dim_id = d_inner_cell_dim_ids[inner_cell_idx * d + l];
                    float left_coor = dim_id * inner_cell_size;
                    float right_coor = (dim_id + 1) * inner_cell_size;
                    left_coor -= x[l];
                    right_coor -= x[l];

                    left_coor *= left_coor;
                    right_coor *= right_coor;
                    if (left_coor < right_coor) {
                        small_dist += left_coor;
                        large_dist += right_coor;
                    } else {
                        small_dist += right_coor;
                        large_dist += left_coor;
                    }
                }
                small_dist = sqrt(small_dist);
                large_dist = sqrt(large_dist);

                bool included = small_dist <= eps;
                bool fully_included = large_dist <= eps;

                const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;
                if (fully_included) {
                    for (int l = 0; l < d; l++) {

                        const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x[l])) -
                                                   (d_sum_cos[inner_cell_idx * d + l] * sin(x[l])));
                        sum[l] += to_be_added;
                    }
                    number_of_neighbors += inner_cell_number_of_points;
                } else if (included) {
                    s_included[included_size] = inner_cell_idx;
                    included_size++;
//                    for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
//                        int q = d_inner_cell_points[q_idx];
//
//                        if (gpu_distance(p, q, d_D_current, d) <= eps) {
//                            for (int l = 0; l < d; l++) {
//                                sum[l] += sin(d_D_current[q * d + l] - x[l]);
//                            }
//                            number_of_neighbors++;
//                        }
//                    }
                }

                /// increment
                inner_cell_idx++;
                group_counter++;
//                }
                if (inner_cell_idx >= outer_cell_end) {
                    loc++;
                    if (loc < pre_end) {
                        outer_cell = d_pre_grid_cells[loc];
                        outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                        outer_cell_end = get_end(d_outer_grid_ends, outer_cell);

                        inner_cell_idx = outer_cell_start;
                    }
                }
            }

            /// look through group
            for (int included_idx = 0; included_idx < included_size; included_idx++) {

                int inner_cell_idx = s_included[included_idx];
                const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;
                for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                    int q = d_inner_cell_points[q_idx];

                    if (gpu_distance(p, q, d_D_current, d) <= eps) {
                        for (int l = 0; l < d; l++) {
                            sum[l] += sin(d_D_current[q * d + l] - x[l]);
                        }
                        number_of_neighbors++;
                    }
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = x[l] + sum[l] / number_of_neighbors;
        }

        if (number_of_neighbors != center_inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
}

__global__ void
kernel_itr_grid_STAD5_5_2(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    extern __shared__ float s_d[];
    float *sum = &s_d[threadIdx.x * d];


    const int center_inner_cell_idx = blockIdx.x;
    const int center_inner_cell_start = get_start(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_end = get_end(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_size = center_inner_cell_end - center_inner_cell_start;

    int iterations = center_inner_cell_size / blockDim.x;
    if (center_inner_cell_size % blockDim.x) iterations++;

    for (int itr = 0; itr < iterations; itr++) {
        int p = 0;
        int p_idx = itr * blockDim.x + threadIdx.x + center_inner_cell_start;
        if (p_idx < center_inner_cell_end) {
            p = d_inner_cell_points[p_idx];

            const int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                       outer_cell_size);

            for (int l = 0; l < d; l++) {
                sum[l] = 0.;
            }
            int number_of_neighbors = 0;

            const int pre_start = get_start(d_pre_grid_ends, tmp_outer_cell);
            const int pre_end = get_end(d_pre_grid_ends, tmp_outer_cell);

            for (int loc = pre_start; loc < pre_end; loc++) {
                const int outer_cell = d_pre_grid_cells[loc];

                const int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                const int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {

                    ///compute included
                    bool included = false;
                    bool fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                        float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];

                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps) {

                        included = true;
                    }

                    if (large_dist <= eps) {
                        fully_included = true;
                    }
//
                    const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                    const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                    const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;
                    if (fully_included) {
                        for (int l = 0; l < d; l++) {
                            const float x_t = d_D_current[p * d + l];
                            const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                       (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                            sum[l] += to_be_added;
                        }
                        number_of_neighbors += inner_cell_number_of_points;
                    } else if (included) {

                        for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                            int q = d_inner_cell_points[q_idx];

                            if (gpu_distance(p, q, d_D_current, d) <= eps) {
                                for (int l = 0; l < d; l++) {
                                    sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                                }
                                number_of_neighbors++;
                            }
                        }
                    }
                }
            }

            int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
            int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        outer_cell, p,
                                                        d, inner_grid_width, inner_cell_size);
            int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
            int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

            for (int l = 0; l < d; l++) {
                d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            }

            if (number_of_neighbors != inner_cell_number_of_points) {
                d_r_local[0] = 0.;
            }
        }
    }
//    delete sum;
}

__global__ void
kernel_itr_grid_STAD5_6(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                        const int *__restrict__ d_pre_inner_grid_fully_cells,
                        const int *__restrict__ d_pre_inner_grid_fully_ends,
                        float *__restrict__ d_r_local,
                        const int *__restrict__ d_outer_grid_ends,
                        const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                        const int *__restrict__ d_inner_cell_dim_ids,
                        float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                        const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                        const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                        const int inner_grid_width, const float inner_cell_size,
                        const int n, const int d, const float eps) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = new float[d]; //todo note: not a good strategy to remove sum in local memory

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        const int p = d_inner_cell_points[p_idx];

        const int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                   outer_cell_size);
        const int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                              tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);


//        for (int l = 0; l < d; l++) {
//            d_D_next[p * d + l] = 0.;
//        }
        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        int number_of_neighbors = 0;

        const int pre_start = get_start(d_pre_grid_ends, tmp_inner_cell_idx);
        const int pre_end = get_end(d_pre_grid_ends, tmp_inner_cell_idx);
        for (int loc = pre_start; loc < pre_end; loc++) {

            //todo idea: these look ups could be avoided by flattining it out beforehand - not sure it will give anything
            const int inner_cell_idx = d_pre_grid_cells[loc];
            const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);

            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                int q = d_inner_cell_points[q_idx];

                if (gpu_distance(p, q, d_D_current, d) <= eps) {
                    for (int l = 0; l < d; l++) {
                        sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
//                        d_D_next[p * d + l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                    }
                    number_of_neighbors++;
                }
            }
        }

        const int pre_fully_start = get_start(d_pre_inner_grid_fully_ends, tmp_inner_cell_idx);
        const int pre_fully_end = get_end(d_pre_inner_grid_fully_ends, tmp_inner_cell_idx);
        for (int loc = pre_fully_start; loc < pre_fully_end; loc++) {
            const int inner_cell_idx = d_pre_inner_grid_fully_cells[loc];

            const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

            for (int l = 0; l < d; l++) {
                const float x_t = d_D_current[p * d + l];
                const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                sum[l] += to_be_added;
//                d_D_next[p * d + l] += to_be_added;
            }
            number_of_neighbors += inner_cell_number_of_points;
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
//            d_D_next[p * d + l] /= number_of_neighbors;
//            d_D_next[p * d + l] += d_D_current[p * d + l];
        }

        if (number_of_neighbors != inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
    delete sum;
}


__global__ void
kernel_itr_grid_STAD5_6_1(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          const int *__restrict__ d_pre_inner_grid_fully_cells,
                          const int *__restrict__ d_pre_inner_grid_fully_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    extern __shared__ float s_d[];
//    float *sum = new float[d]; //todo note: not a good strategy to remove sum in local memory
    float *sum = &s_d[threadIdx.x * d]; //todo note: not a good strategy to remove sum in local memory

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        const int p = d_inner_cell_points[p_idx];

        const int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                   outer_cell_size);
        const int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                              tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);


//        for (int l = 0; l < d; l++) {
//            d_D_next[p * d + l] = 0.;
//        }
        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        int number_of_neighbors = 0;

        //todo idea: maybe try to load points into shared memory and run the points in the same inner grid in the same block...
        const int pre_start = get_start(d_pre_grid_ends, tmp_inner_cell_idx);
        const int pre_end = get_end(d_pre_grid_ends, tmp_inner_cell_idx);
        for (int loc = pre_start; loc < pre_end; loc++) {

            //todo idea: these look ups could be avoided by flattining it out beforehand - not sure it will give anything
//            const int inner_cell_idx = d_pre_grid_cells[loc];
//            const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
//            const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
//
//            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
//                int q = d_inner_cell_points[q_idx];
            int q = d_pre_grid_cells[loc];

            if (gpu_distance(p, q, d_D_current, d) <= eps) {
                for (int l = 0; l < d; l++) {
                    sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
//                        d_D_next[p * d + l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                }
                number_of_neighbors++;
            }
//            }
        }

        const int pre_fully_start = get_start(d_pre_inner_grid_fully_ends, tmp_inner_cell_idx);
        const int pre_fully_end = get_end(d_pre_inner_grid_fully_ends, tmp_inner_cell_idx);
        for (int loc = pre_fully_start; loc < pre_fully_end; loc++) {
            const int inner_cell_idx = d_pre_inner_grid_fully_cells[loc];

            const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

            for (int l = 0; l < d; l++) {
                const float x_t = d_D_current[p * d + l];
                const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                sum[l] += to_be_added;
//                d_D_next[p * d + l] += to_be_added;
            }
            number_of_neighbors += inner_cell_number_of_points;
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
//            d_D_next[p * d + l] /= number_of_neighbors;
//            d_D_next[p * d + l] += d_D_current[p * d + l];
        }

        if (number_of_neighbors != inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
//    delete sum;
}

__global__ void
kernel_itr_grid_STAD5_6_2(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          const int *__restrict__ d_pre_inner_grid_fully_cells,
                          const int *__restrict__ d_pre_inner_grid_fully_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps) {

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = new float[d * 2];
    float *x = &sum[d];

    const int center_inner_cell_idx = blockIdx.x;
    const int center_inner_cell_start = get_start(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_end = get_end(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_size = center_inner_cell_end - center_inner_cell_start;
    const int pre_start = get_start(d_pre_grid_ends, center_inner_cell_idx);
    const int pre_end = get_end(d_pre_grid_ends, center_inner_cell_idx);
    const int pre_fully_start = get_start(d_pre_inner_grid_fully_ends, center_inner_cell_idx);
    const int pre_fully_end = get_end(d_pre_inner_grid_fully_ends, center_inner_cell_idx);

    for (int p_idx = threadIdx.x + center_inner_cell_start; p_idx < center_inner_cell_end; p_idx += blockDim.x) {
        const int p = d_inner_cell_points[p_idx];

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
            x[l] = d_D_current[p * d + l];
        }
        int number_of_neighbors = 0;

        for (int loc = pre_start; loc < pre_end; loc++) {
            int q = d_pre_grid_cells[loc];
            if (gpu_distance(p, q, d_D_current, d) <= eps) {
                for (int l = 0; l < d; l++) {
                    sum[l] += sin(d_D_current[q * d + l] - x[l]);
                }
                number_of_neighbors++;
            }
        }


        for (int loc = pre_fully_start; loc < pre_fully_end; loc++) {
            const int inner_cell_idx = d_pre_inner_grid_fully_cells[loc];

            const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
            const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

            for (int l = 0; l < d; l++) {
//                const float x_t = d_D_current[p * d + l];
                const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x[l])) -
                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x[l])));
                sum[l] += to_be_added;
            }
            number_of_neighbors += inner_cell_number_of_points;
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = x[l] + sum[l] / number_of_neighbors;
        }

        if (number_of_neighbors != center_inner_cell_size) {
            d_r_local[0] = 0.;
        }
    }
    delete sum;
//    delete x;
}


__global__ void
kernel_itr_grid_STAD5_6_3(const int *__restrict__ d_pre_grid_cells, const int *__restrict__ d_pre_grid_ends,
                          const int *__restrict__ d_pre_inner_grid_fully_cells,
                          const int *__restrict__ d_pre_inner_grid_fully_ends,
                          float *__restrict__ d_r_local,
                          const int *__restrict__ d_outer_grid_ends,
                          const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                          const int *__restrict__ d_inner_cell_dim_ids,
                          float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                          const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d, const float eps) {

    extern __shared__ float s_data[];

    const int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = &s_data[blockDim.x * d + threadIdx.x * d * 2];
    float *x = &sum[d];
//    float *sum = new float[d * 2];
//    float *x = &sum[d];

    const int center_inner_cell_idx = blockIdx.x;
    const int center_inner_cell_start = get_start(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_end = get_end(d_inner_grid_ends, center_inner_cell_idx);
    const int center_inner_cell_size = center_inner_cell_end - center_inner_cell_start;
    const int pre_start = get_start(d_pre_grid_ends, center_inner_cell_idx);
    const int pre_end = get_end(d_pre_grid_ends, center_inner_cell_idx);
    const int pre_fully_start = get_start(d_pre_inner_grid_fully_ends, center_inner_cell_idx);
    const int pre_fully_end = get_end(d_pre_inner_grid_fully_ends, center_inner_cell_idx);

    int iterations = center_inner_cell_size / blockDim.x;
    if (center_inner_cell_size % blockDim.x) iterations++;

    //for (int p_idx = threadIdx.x + center_inner_cell_start; p_idx < center_inner_cell_end; p_idx += blockDim.x) {
    for (int itr = 0; itr < iterations; itr++) {
        int number_of_neighbors = 0;
        int p = 0;
        int p_idx = itr * blockDim.x + threadIdx.x + center_inner_cell_start;
        if (p_idx < center_inner_cell_end) {
            p = d_inner_cell_points[p_idx];
            for (int l = 0; l < d; l++) {
                sum[l] = 0.;
                x[l] = d_D_current[p * d + l];
            }
        }

        for (int offset = pre_start; offset < pre_end; offset += blockDim.x) {
            if (offset + threadIdx.x < pre_end) {
                int q = d_pre_grid_cells[offset + threadIdx.x];
                for (int l = 0; l < d; l++) {
                    s_data[threadIdx.x * d + l] = d_D_current[q * d + l];
                }
            }
            __syncthreads();
            if (p_idx < center_inner_cell_end) {
                for (int loc = 0; loc < blockDim.x; loc++) {

                    if (offset + loc < pre_end) {

                        float dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float diff = s_data[loc * d + l] - x[l];
                            dist += diff * diff;
                        }
                        dist = sqrt(dist);

                        if (dist <= eps) {
                            for (int l = 0; l < d; l++) {
                                sum[l] += sin(s_data[loc * d + l] - x[l]);
                            }
                            number_of_neighbors++;
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (p_idx < center_inner_cell_end) {
            for (int loc = pre_fully_start; loc < pre_fully_end; loc++) {
                const int inner_cell_idx = d_pre_inner_grid_fully_cells[loc];

                const int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                const int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                for (int l = 0; l < d; l++) {
//                const float x_t = d_D_current[p * d + l];
                    const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x[l])) -
                                               (d_sum_cos[inner_cell_idx * d + l] * sin(x[l])));
                    sum[l] += to_be_added;
                }
                number_of_neighbors += inner_cell_number_of_points;
            }

            for (int l = 0; l < d; l++) {
                d_D_next[p * d + l] = x[l] + sum[l] / number_of_neighbors;
            }

            if (d_r_local[0] != 0. && number_of_neighbors != center_inner_cell_size) {
                d_r_local[0] = 0.;
            }
        }
    }
//    delete sum;
//    delete x;
}

/**
 * for n=100,000 d=4 registres and memory though put is the bottleneck
 * in this version i try to reduce the used registres
 */
__global__ void
kernel_itr_grid_STAD5_4(float *__restrict__ d_r_local,
                        const int *__restrict__ d_outer_grid_ends,
                        const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                        const int *__restrict__ d_inner_cell_dim_ids,
                        float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                        const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                        const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                        const int inner_grid_width, const float inner_cell_size,
                        const int n, const int d, const float eps) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = new float[d];
    unsigned char *tmp = new unsigned char[outer_grid_dims];
    unsigned char *idxs = new unsigned char[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        int p = d_inner_cell_points[p_idx];

        /*
        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);
        */

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        float r_c = 0.;
        int number_of_neighbors = 0;

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps) {
                            included = true;
                        }

                        if (large_dist <= eps) {
                            fully_included = true;
                        }

                        if (fully_included) {
                            for (int l = 0; l < d; l++) {
                                const float x_t = d_D_current[p * d + l];
                                /*const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                                sum[l] += to_be_added;*/

                                sum[l] += ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                           (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                            }
                            number_of_neighbors += inner_cell_number_of_points;
                        } else if (included) {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];

                                if (gpu_distance(p, q, d_D_current, d) <= eps) {
                                    for (int l = 0; l < d; l++) {
                                        sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                                    }
                                    number_of_neighbors++;
                                }
                            }
                        }
                    }
                }
            }
            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        if (number_of_neighbors != inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

//same as grid_STAD5 but with different blocks / threads
__global__ void
kernel_itr_grid_STAD5_3(float *__restrict__ d_r_local,
                        const int *__restrict__ d_outer_grid_ends,
                        const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                        const int *__restrict__ d_inner_cell_dim_ids,
                        float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                        const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                        const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                        const int inner_grid_width, const float inner_cell_size,
                        const int n, const int d, const float eps) {

    int outer_grid_radius = ceil(eps / outer_cell_size);
    __shared__ int number_of_neighbors;

    __shared__ float *sum;
    sum = new float[d];
    __syncthreads();
    int *tmp = new int[outer_grid_dims];
    int *idxs = new int[outer_grid_dims];

    for (int p_idx = blockIdx.x; p_idx < n; p_idx += gridDim.x) {
        int p = d_inner_cell_points[p_idx];

        const int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims,
                                                   outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        float r_c = 0.;
        number_of_neighbors = 0;
        __syncthreads();

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps) {

                            included = true;
                        }

                        if (large_dist <= eps) {
                            fully_included = true;
                        }

                        if (fully_included) {
                            if (threadIdx.x == 0) {
                                for (int l = 0; l < d; l++) {
                                    const float x_t = d_D_current[p * d + l];
                                    const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                               (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                                    atomicAdd(&sum[l], to_be_added);
                                }
                                atomicAdd(&number_of_neighbors, inner_cell_number_of_points);
                            }
                        } else if (included) {
                            for (int q_idx = inner_cell_start + threadIdx.x;
                                 q_idx < inner_cell_end; q_idx += blockDim.x) {
                                int q = d_inner_cell_points[q_idx];

                                if (gpu_distance(p, q, d_D_current, d) <= eps) {
                                    for (int l = 0; l < d; l++) {
                                        atomicAdd(&sum[l], sin(d_D_current[q * d + l] - d_D_current[p * d + l]));
                                    }
                                    atomicInc((unsigned int *) &number_of_neighbors, n);
                                }
                            }
                        }
                    }
                }
            }
            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
            int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        outer_cell, p,
                                                        d, inner_grid_width, inner_cell_size);
            int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
            int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
            int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

            for (int l = 0; l < d; l++) {
                d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
            }

            if (number_of_neighbors != inner_cell_number_of_points) {
                d_r_local[0] = 0.;
            }
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

__global__ void
kernel_itr_grid_STAD5_2(float *__restrict__ d_r_local,
                        const int *__restrict__ d_outer_grid_ends,
                        const int *__restrict__ d_inner_grid_ends, const int *__restrict__ d_inner_cell_points,
                        const int *__restrict__ d_inner_cell_dim_ids,
                        float *__restrict__ d_D_next, const float *__restrict__ d_D_current,
                        const float *__restrict__ d_sum_sin, const float *__restrict__ d_sum_cos,
                        const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                        const int inner_grid_width, const float inner_cell_size,
                        const int n, const int d, const float eps) {

    int outer_grid_radius = ceil(eps / outer_cell_size);

    float *sum = new float[d];
    int *tmp = new int[outer_grid_dims];
    int *idxs = new int[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {
        int p = d_inner_cell_points[p_idx];

        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        for (int l = 0; l < d; l++) {
            sum[l] = 0.;
        }
        float r_c = 0.;
        int number_of_neighbors = 0;

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        bool finished = false;
        while (!finished) {
            //while(idxs[outer_grid_dims - 1] <= outer_grid_radius) {
            bool included = false;
            bool fully_included = false;
            int inner_cell_start = 0;
            int inner_cell_end = 0;
            int inner_cell_number_of_points = 0;
            while (!finished && (!included || fully_included || inner_cell_number_of_points == 0)) {

                ///check within bounds
                bool within_bounds = true;
                for (int l = 0; l < outer_grid_dims; l++) {
                    if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                        within_bounds = false;
                    }
                }

                if (within_bounds) {
                    ///check size
                    int outer_number_of_cells = 1;
                    int outer_cell = 0;
                    for (int l = 0; l < outer_grid_dims; l++) {
                        outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                        outer_number_of_cells *= outer_grid_width;
                    }

                    int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                    int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                    int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                    if (outer_cell_number_of_cells > 0) {
                        for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                            inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                            inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                            inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                            ///compute included
                            included = false;
                            fully_included = false;

                            float small_dist = 0.;
                            float large_dist = 0.;
                            for (int l = 0; l < d; l++) {
                                float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                                float right_coor =
                                        (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                                left_coor -= d_D_current[p * d + l];
                                right_coor -= d_D_current[p * d + l];

                                left_coor *= left_coor;
                                right_coor *= right_coor;
                                if (left_coor < right_coor) {
                                    small_dist += left_coor;
                                    large_dist += right_coor;
                                } else {
                                    small_dist += right_coor;
                                    large_dist += left_coor;
                                }
                            }
                            small_dist = sqrt(small_dist);
                            large_dist = sqrt(large_dist);

                            if (small_dist <= eps) {
                                included = true;
                            }

                            if (large_dist <= eps) {
                                fully_included = true;
                            }

                            if (fully_included) {
                                for (int l = 0; l < d; l++) {
                                    const float x_t = d_D_current[p * d + l];
                                    const float to_be_added = ((d_sum_sin[inner_cell_idx * d + l] * cos(x_t)) -
                                                               (d_sum_cos[inner_cell_idx * d + l] * sin(x_t)));
                                    sum[l] += to_be_added;
                                }
                                number_of_neighbors += inner_cell_number_of_points;
                            } else if (included) {
                                /*for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                    int q = d_inner_cell_points[q_idx];

                                    if (gpu_distance(p, q, d_D_current, d) <= eps) {
                                        for (int l = 0; l < d; l++) {
                                            sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                                        }
                                        number_of_neighbors++;
                                    }
                                }*/
                            }
                        }
                    }
                }
                ///increment index
                idxs[0]++;
                int l = 1;
                while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                    idxs[l - 1] = -outer_grid_radius;
                    idxs[l]++;
                    l++;
                }

                if (idxs[outer_grid_dims - 1] > outer_grid_radius) {
                    finished = true;
                }
            }

            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                int q = d_inner_cell_points[q_idx];

                if (gpu_distance(p, q, d_D_current, d) <= eps) {
                    for (int l = 0; l < d; l++) {
                        sum[l] += sin(d_D_current[q * d + l] - d_D_current[p * d + l]);
                    }
                    number_of_neighbors++;
                }
            }
        }

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);
        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = d_D_current[p * d + l] + sum[l] / number_of_neighbors;
        }

        if (number_of_neighbors != inner_cell_number_of_points) {
            d_r_local[0] = 0.;
        }
    }
    delete sum;
    delete tmp;
    delete idxs;
}

//        while (true) {
//
//            if (q_idx >= inner_cell_end) {
//                while (inner_cell_idx >= outer_cell_end) {
//                    int outer_grid_start = 0;
//                    int outer_grid_end = 0;
//                    int outer_grid_size = 0;
//                    do {
//                        //go though outer grid cells
//                        idxs[0]++
//                        int l = 1;
//                        while (l < outer_grid_dims && idxs[l - 1] > radius) {
//                            idxs[l - 1] = -radius;
//                            idxs[l]++;
//                            l++;
//                        }
//                        if (idxs[outer_grid_dims - 1] > radius) {
//                            break;
//                        }
//
//
//                        //todo this could be out of bound (tmp[l] + idxs[l])
//
//                        //find next outer cell
//                        number_of_cells = 1;
//                        outer_cell = 0;
//                        for (int l = 0; l < outer_grid_dims; l++) {
//                            outer_cell += (tmp[l] + idxs[l]) * number_of_cells;
//                            number_of_cells *= outer_width;
//                        }
//
//                        outer_grid_start = get_start(d_outer_grid_ends, outer_cell);
//                        outer_grid_end = get_end(d_outer_grid_ends, outer_cell);
//                        outer_grid_size = outer_grid_end - outer_grid_start;
//
//                    } while (outer_grid_size == 0 && idxs[outer_grid_dims - 1] <= radius);
//
//
//                    ///...
//
//                    bool included = false;
//                    fully_included = false;
//                    inner_cell_idx = outer_cell_start;
//                    do {
//                        for (int l = 0; l < d; l++) {
//                            float left_coor = (d_inner_grid_cell_ids[??]) *cell_size;
//                            float right_coor = (d_inner_grid_cell_ids[??] +1) *cell_size;
//                            left_coor -= d_D_current[p * d + l];
//                            right_coor -= d_D_current[p * d + l];
//                            left_coor *= left_coor;
//                            right_coor *= right_coor;
//                            if (left_coor < right_coor) {
//                                small_dist += left_coor;
//                                large_dist += right_coor;
//                            } else {
//                                small_dist += right_coor;
//                                large_dist += left_coor;
//                            }
//                            small_dist = sqrt(small_dist);
//                            large_dist = sqrt(large_dist);
//                        }
//
//                        if (small_dist <= eps) {
//                            included = true;
//                        }
//
//                        if (large_dist <= eps) {
//                            fully_included = true;
//                        }
//
//
//                        inner_grid_start = get_start(d_inner_grid_ends, inner_cell_idx);
//                        inner_grid_end = get_end(d_inner_grid_ends, inner_cell_idx);
//                        inner_grid_size = inner_grid_end - inner_grid_start;
//
//                    } while (inner_grid_size == 0 && );
//                }
//
//                ///...
//                q_idx = inner_cell_start;
//            }
//        }

//__global__
//void
//kernel_itr_grid_STAD5_simple(const float *__restrict__ d_sum_cos, const float *__restrict__ d_sum_sin,
//                             float *__restrict__ d_r_local,
//                             const int *__restrict__ d_grid, const int *__restrict__ d_grid_ends,
//                             const float *__restrict__ d_D_current, float *__restrict__ d_D_next, const int n,
//                             const int d,
//                             const float eps, const int width, const int grid_dims, const float cell_size,
//                             int number_of_cells, int itr) {
//
//    float *sum = new float[d];
//    int *tmp = new int[d];//todo only needs to be outer_grid_dims
//    int *idxs = new int[d];
//
//    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
//
//        while (idxs[outer_grid_dims - 1] > radius) {
//            idxs[0]++
//            int l = 1;
//            while (l < outer_grid_dims && idxs[l - 1] > radius) {
//                idxs[l - 1] = -radius;
//                idxs[l]++;
//                l++;
//            }
//
//            number_of_cells = 1;
//            outer_cell = 0;
//            for (int l = 0; l < outer_grid_dims; l++) {
//                outer_cell += (tmp[l] + idxs[l]) * number_of_cells;
//                number_of_cells *= outer_width;
//            }
//
//            outer_grid_start = get_start(d_outer_grid_ends, outer_cell);
//            outer_grid_end = get_end(d_outer_grid_ends, outer_cell);
//            outer_grid_size = outer_grid_end - outer_grid_start;
//
//            for (int inner_cell_location = outer_grid_start;
//                 inner_cell_location < outer_grid_end; inner_cell_location++) {
//                inner_grid_start = get_start(d_inner_grid_ends, inner_cell_location);
//                inner_grid_end = get_end(d_inner_grid_ends, inner_cell_location);
//                inner_grid_size = inner_grid_end - inner_grid_start;
//
//                bool included = false;
//                bool fully_included = false;
//                for (int l = 0; l < d; l++) {
//                    float left_coor = (d_inner_grid_cell_ids[??]) *cell_size;
//                    float right_coor = (d_inner_grid_cell_ids[??] +1) *cell_size;
//                    left_coor -= d_D_current[p * d + l];
//                    right_coor -= d_D_current[p * d + l];
//                    left_coor *= left_coor;
//                    right_coor *= right_coor;
//                    if (left_coor < right_coor) {
//                        small_dist += left_coor;
//                        large_dist += right_coor;
//                    } else {
//                        small_dist += right_coor;
//                        large_dist += left_coor;
//                    }
//                    small_dist = sqrt(small_dist);
//                    large_dist = sqrt(large_dist);
//                }
//
//                if (small_dist <= eps) {
//                    included = true;
//                }
//
//                if (large_dist <= eps) {
//                    fully_included = true;
//                }
//
//                if (fully_included) {
//
//                } else if (included) {
//                    for (int q_idx = inner_grid_start; q_idx < inner_grid_end; q_idx++) {
//                        int q = d_inner_grid_points[q_idx];
//
//
//                    }
//                }
//            }
//        }
//    }
//    delete sum;
//    delete tmp;
//    delete idxs;
//}

void build_the_grid(int *&d_outer_grid_sizes, int *&d_outer_grid_ends, int *&d_new_outer_grid_ends,
                    int *&d_inner_grid_sizes, int *&d_inner_grid_ends, int *&d_new_inner_grid_ends,
                    int *&d_inner_grid_included, int *&d_inner_grid_idxs,
                    int *&d_inner_grid_points, int *&d_inner_grid_cell_dim_ids, int *&d_new_inner_grid_cell_dim_ids,
                    float *&d_D_current, float *&d_sum_sin, float *&d_sum_cos,
                    int n, int d,
                    int outer_grid_number_of_cells, int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                    int inner_grid_width, int inner_grid_dims, float inner_cell_size) {

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    int outer_grid_umber_of_blocks = outer_grid_number_of_cells / BLOCK_SIZE;
    if (outer_grid_number_of_cells % BLOCK_SIZE)
        outer_grid_umber_of_blocks++;

    /// building the grid
    //initializing the grid
    gpu_set_all_zero(d_outer_grid_sizes, outer_grid_number_of_cells);
    gpu_set_all_zero(d_outer_grid_ends, outer_grid_number_of_cells);
    gpu_set_all_zero(d_inner_grid_sizes, n);
    gpu_set_all_zero(d_inner_grid_ends, n);
    gpu_set_all_zero(d_inner_grid_included, n);
    gpu_set_all_zero(d_inner_grid_idxs, n);

    // 1. compute the number of points in each outer grid cell
    kernel_outer_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_outer_grid_sizes, d_D_current, n, d,
            outer_grid_width, outer_grid_dims,
            outer_cell_size);

    // 2. inclusive scan of d_outer_grid_sizes
    inclusive_scan(d_outer_grid_sizes, d_outer_grid_ends, outer_grid_number_of_cells);

    //reset outer_sizes
    gpu_set_all_zero(d_outer_grid_sizes, outer_grid_number_of_cells);

    // 3. compute the id of the inner cell it belongs to and write that in the location of that point inside the outer grid cell
    kernel_inner_grid_marking << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                     (d_inner_grid_cell_dim_ids, d_outer_grid_sizes,
                                                             d_outer_grid_ends,
                                                             d_D_current,
                                                             n, d,
                                                             outer_grid_width, outer_grid_dims,
                                                             outer_cell_size,
                                                             inner_grid_width, inner_grid_dims,
                                                             inner_cell_size);

    // 4. compute the size of each inner grid cell.
    kernel_inner_grid_sizes << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                   //<< < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                   (d_inner_grid_sizes, d_inner_grid_included, d_inner_grid_cell_dim_ids,
                                                           d_outer_grid_ends,
                                                           d_D_current, n, d,
                                                           outer_grid_width, outer_grid_dims, outer_cell_size,
                                                           inner_grid_width, inner_grid_dims, inner_cell_size);

    // 5. inclusive scan of d_inner_grid_sizes
    inclusive_scan(d_inner_grid_sizes, d_inner_grid_ends, n);

    // 6. inclusive scan of d_inner_grid_included
    inclusive_scan(d_inner_grid_included, d_inner_grid_idxs, n);

    // reset d_inner_grid_sizes
    gpu_set_all_zero(d_inner_grid_sizes, n);

    // 7. populate d_inner_grid_points
    kernel_inner_grid_populate << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_inner_grid_points, d_inner_grid_sizes,
            d_inner_grid_cell_dim_ids,
            d_inner_grid_ends,
            d_outer_grid_ends,
            d_D_current,
            n, d,
            outer_grid_width, outer_grid_dims,
            outer_cell_size,
            inner_grid_width, inner_grid_dims,
            inner_cell_size);

    // 8. repack inner_grid_ends! d_new_inner_grid_ends[inner_grid_idx] = inner_grid_end;
    kernel_inner_grid_repack << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_new_inner_grid_ends,
            d_new_inner_grid_cell_dim_ids, d_inner_grid_idxs,
            d_inner_grid_included, d_inner_grid_ends,
            d_inner_grid_cell_dim_ids,
            n, d);

    // 9. repack outer_grid_ends! d_new_outer_grid_ends[outer_cell] = inner_grid_idx;
    kernel_outer_grid_repack << < outer_grid_umber_of_blocks, min(outer_grid_number_of_cells, BLOCK_SIZE) >> > (
            d_new_outer_grid_ends, d_outer_grid_ends, d_inner_grid_idxs,
                    outer_grid_number_of_cells);

    swap(d_new_inner_grid_cell_dim_ids, d_inner_grid_cell_dim_ids);
    swap(d_new_inner_grid_ends, d_inner_grid_ends);
    swap(d_new_outer_grid_ends, d_outer_grid_ends);
}

__global__ void
GPU_synCluster_float_grid(int *__restrict__ d_C, const float *__restrict__ d_D_current,
                          int *d_outer_grid_ends,
                          int *__restrict__ d_inner_cell_dim_ids, int *__restrict__ d_inner_cell_ends,
                          const int outer_grid_width, const int outer_grid_dims, const float outer_cell_size,
                          const int inner_grid_width, const float inner_cell_size,
                          const int n, const int d) {

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {

        int outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends, outer_cell, p,
                                                    d, inner_grid_width, inner_cell_size);

        int inner_cell_start = get_start(d_inner_cell_ends, inner_cell_idx);
        int inner_cell_end = get_end(d_inner_cell_ends, inner_cell_idx);
        int cell_number_of_points = inner_cell_end - inner_cell_start;

        if (cell_number_of_points > 1) {
            d_C[p] = inner_cell_idx;
        }
    }
}

__device__ bool MBR_in_eps(float *mbr, int *tmp, int *idxs,
                           int *d_outer_grid_ends,
                           int *d_inner_grid_ends, int *d_inner_cell_points, int *d_inner_cell_dim_ids,
                           float *d_D_current,
                           int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                           float inner_cell_size,
                           int d, float eps, int p_mark, int p) {

    int outer_grid_radius = ceil((eps / 2.) / outer_cell_size);

    for (int l = 0; l < d; l++) {
        mbr[l] = d_D_current[p * d + l];
    }

    for (int l = 0; l < outer_grid_dims; l++) {
        float val = d_D_current[p * d + l];
        tmp[l] = val / outer_cell_size;
        if (tmp[l] == outer_grid_width) {
            tmp[l]--;
        }
    }

    for (int l = 0; l < outer_grid_dims; l++) {
        idxs[l] = -outer_grid_radius;
    }

    while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

        ///check within bounds
        bool within_bounds = true;
        for (int l = 0; l < outer_grid_dims; l++) {
            if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                within_bounds = false;
            }
        }

        if (within_bounds) {
            ///check size
            int outer_number_of_cells = 1;
            int outer_cell = 0;
            for (int l = 0; l < outer_grid_dims; l++) {
                outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                outer_number_of_cells *= outer_grid_width;
            }

            int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
            int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
            int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
            if (outer_cell_number_of_cells > 0) {
                for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                    int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                    int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                    int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                    ///compute included
                    bool included = false;
                    bool fully_included = false;

                    float small_dist = 0.;
                    float large_dist = 0.;
                    for (int l = 0; l < d; l++) {
                        float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                        float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                        left_coor -= d_D_current[p * d + l];
                        right_coor -= d_D_current[p * d + l];

                        left_coor *= left_coor;
                        right_coor *= right_coor;
                        if (left_coor < right_coor) {
                            small_dist += left_coor;
                            large_dist += right_coor;
                        } else {
                            small_dist += right_coor;
                            large_dist += left_coor;
                        }
                    }
                    small_dist = sqrt(small_dist);
                    large_dist = sqrt(large_dist);

                    if (small_dist <= eps / 2) {

                        included = true;
                    }

                    if (included) {
                        for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                            int q = d_inner_cell_points[q_idx];

                            if (gpu_distance(p, q, d_D_current, d) <= eps / 2) {

                                for (int l = 0; l < d; l++) {
                                    float old_dist = mbr[l] - d_D_current[p_mark * d + l];
                                    old_dist *= old_dist;
                                    float new_dist = d_D_current[q * d + l] - d_D_current[p_mark * d + l];
                                    new_dist *= new_dist;
                                    if (new_dist < old_dist) {
                                        mbr[l] = d_D_current[q * d + l];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ///increment index
        idxs[0]++;
        int l = 1;
        while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
            idxs[l - 1] = -outer_grid_radius;
            idxs[l]++;
            l++;
        }
    }

    float dist_to_MBR = 0.;
    for (int l = 0; l < d; l++) {
        float dist = mbr[l] - d_D_current[p_mark * d + l];
        dist *= dist;
        dist_to_MBR += dist;
    }
    return sqrt(dist_to_MBR) <= eps;
}

__global__ void kernel_float_grid_extra_check(float *d_r_local,
                                              int *d_outer_grid_ends,
                                              int *d_inner_grid_ends, int *d_inner_cell_points,
                                              int *d_inner_cell_dim_ids,
                                              float *d_D_current, float *d_sum_sin, float *d_sum_cos,
                                              int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                                              int inner_grid_width, float inner_cell_size,
                                              int n, int d, float eps, float eps_extra) {


    int outer_grid_radius = ceil(eps_extra / outer_cell_size);

    extern __shared__ float s_d[];

    float *mbr = &s_d[threadIdx.x * (d + 4 * outer_grid_dims)];
    int *tmp = (int *) &mbr[d];
    int *idxs = &tmp[outer_grid_dims];

    int *tmp_extra = &idxs[outer_grid_dims];
    int *idxs_extra = &tmp_extra[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {

        if (d_r_local[0] == 0.) {

            return;
        }

        int p = d_inner_cell_points[p_idx];

        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps_extra) {

                            included = true;
                        }

                        if (large_dist <= eps) {
                            //not eps_extra since we need to check all points within the extra range
                            fully_included = true;
                        }

                        if (!fully_included && included) {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];

                                float dist = gpu_distance(p, q, d_D_current, d);

                                if (eps < dist && dist <= eps_extra) {

                                    //printf("candidate q: %d found at dist: %f\n", q, dist);

                                    if (MBR_in_eps(mbr, tmp_extra, idxs_extra, d_outer_grid_ends,
                                                   d_inner_grid_ends, d_inner_cell_points, d_inner_cell_dim_ids,
                                                   d_D_current,
                                                   outer_grid_width, outer_grid_dims, outer_cell_size,
                                                   inner_cell_size,
                                                   d, eps, p, q)) {
                                        d_r_local[0] = 0.;

                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }
}


__global__ void kernel_float_grid_extra_check_part1(int *d_to_be_checked_size, int *d_to_be_checked,
                                                    float *d_r_local,
                                                    int *d_outer_grid_ends,
                                                    int *d_inner_grid_ends, int *d_inner_cell_points,
                                                    int *d_inner_cell_dim_ids,
                                                    float *d_D_current, float *d_sum_sin, float *d_sum_cos,
                                                    int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                                                    int inner_grid_width, float inner_cell_size,
                                                    int n, int d, float eps, float eps_extra) {


    int outer_grid_radius = ceil(eps_extra / outer_cell_size);

    extern __shared__ float s_d[];

    float *mbr = &s_d[threadIdx.x * (d + 2 * outer_grid_dims)];
    int *tmp = (int *) &mbr[d];
    int *idxs = &tmp[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {

        int p = d_inner_cell_points[p_idx];

        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps_extra) {

                            included = true;
                        }

                        if (large_dist <= eps) {
                            //not eps_extra since we need to check all points within the extra range
                            fully_included = true;
                        }

                        if (!fully_included && included) {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];

                                float dist = gpu_distance(p, q, d_D_current, d);

                                if (eps < dist && dist <= eps_extra) {

                                    int i = atomicInc((unsigned int *) d_to_be_checked_size, n * n);
//                                    if (i > n) {
//                                        printf("not enough space!\n");
//                                    }

                                    d_to_be_checked[i * 2 + 0] = p;
                                    d_to_be_checked[i * 2 + 1] = q;
                                }
                            }
                        }
                    }
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }
}


__global__ void kernel_float_grid_extra_check_part1_count(int *d_to_be_checked_size, int *d_to_be_checked,
                                                          float *d_r_local,
                                                          int *d_outer_grid_ends,
                                                          int *d_inner_grid_ends, int *d_inner_cell_points,
                                                          int *d_inner_cell_dim_ids,
                                                          float *d_D_current, float *d_sum_sin, float *d_sum_cos,
                                                          int outer_grid_width, int outer_grid_dims,
                                                          float outer_cell_size,
                                                          int inner_grid_width, float inner_cell_size,
                                                          int n, int d, float eps, float eps_extra) {


    int outer_grid_radius = ceil(eps_extra / outer_cell_size);

    extern __shared__ float s_d[];

    float *mbr = &s_d[threadIdx.x * (d + 2 * outer_grid_dims)];
    int *tmp = (int *) &mbr[d];
    int *idxs = &tmp[outer_grid_dims];

    for (int p_idx = threadIdx.x + blockIdx.x * blockDim.x; p_idx < n; p_idx += blockDim.x * gridDim.x) {

        int p = d_inner_cell_points[p_idx];

        int tmp_outer_cell = compute_cell_id(d_D_current, p, d, outer_grid_width, outer_grid_dims, outer_cell_size);
        int tmp_inner_cell_idx = compute_inner_cell_idx(d_D_current, d_inner_cell_dim_ids, d_outer_grid_ends,
                                                        tmp_outer_cell, p, d, inner_grid_width, inner_cell_size);

        for (int l = 0; l < outer_grid_dims; l++) {
            float val = d_D_current[p * d + l];
            tmp[l] = val / outer_cell_size;
            if (tmp[l] == outer_grid_width) {
                tmp[l]--;
            }
        }

        for (int l = 0; l < outer_grid_dims; l++) {
            idxs[l] = -outer_grid_radius;
        }

        while (idxs[outer_grid_dims - 1] <= outer_grid_radius) {

            ///check within bounds
            bool within_bounds = true;
            for (int l = 0; l < outer_grid_dims; l++) {
                if (tmp[l] + idxs[l] < 0 || outer_grid_width <= tmp[l] + idxs[l]) {
                    within_bounds = false;
                }
            }

            if (within_bounds) {
                ///check size
                int outer_number_of_cells = 1;
                int outer_cell = 0;
                for (int l = 0; l < outer_grid_dims; l++) {
                    outer_cell += (tmp[l] + idxs[l]) * outer_number_of_cells;
                    outer_number_of_cells *= outer_grid_width;
                }

                int outer_cell_start = get_start(d_outer_grid_ends, outer_cell);
                int outer_cell_end = get_end(d_outer_grid_ends, outer_cell);
                int outer_cell_number_of_cells = outer_cell_end - outer_cell_start;
                if (outer_cell_number_of_cells > 0) {
                    for (int inner_cell_idx = outer_cell_start; inner_cell_idx < outer_cell_end; inner_cell_idx++) {
                        int inner_cell_start = get_start(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_end = get_end(d_inner_grid_ends, inner_cell_idx);
                        int inner_cell_number_of_points = inner_cell_end - inner_cell_start;

                        ///compute included
                        bool included = false;
                        bool fully_included = false;

                        float small_dist = 0.;
                        float large_dist = 0.;
                        for (int l = 0; l < d; l++) {
                            float left_coor = d_inner_cell_dim_ids[inner_cell_idx * d + l] * inner_cell_size;
                            float right_coor = (d_inner_cell_dim_ids[inner_cell_idx * d + l] + 1) * inner_cell_size;
                            left_coor -= d_D_current[p * d + l];
                            right_coor -= d_D_current[p * d + l];

                            left_coor *= left_coor;
                            right_coor *= right_coor;
                            if (left_coor < right_coor) {
                                small_dist += left_coor;
                                large_dist += right_coor;
                            } else {
                                small_dist += right_coor;
                                large_dist += left_coor;
                            }
                        }
                        small_dist = sqrt(small_dist);
                        large_dist = sqrt(large_dist);

                        if (small_dist <= eps_extra) {

                            included = true;
                        }

                        if (large_dist <= eps) {
                            //not eps_extra since we need to check all points within the extra range
                            fully_included = true;
                        }

                        if (!fully_included && included) {
                            for (int q_idx = inner_cell_start; q_idx < inner_cell_end; q_idx++) {
                                int q = d_inner_cell_points[q_idx];

                                float dist = gpu_distance(p, q, d_D_current, d);

                                if (eps < dist && dist <= eps_extra) {

                                    int i = atomicInc((unsigned int *) d_to_be_checked_size, n * n);

                                }
                            }
                        }
                    }
                }
            }

            ///increment index
            idxs[0]++;
            int l = 1;
            while (l < outer_grid_dims && idxs[l - 1] > outer_grid_radius) {
                idxs[l - 1] = -outer_grid_radius;
                idxs[l]++;
                l++;
            }
        }
    }
}

__global__ void kernel_float_grid_extra_check_part2(int *d_to_be_checked_size, int *d_to_be_checked, float *d_r_local,
                                                    int *d_outer_grid_ends,
                                                    int *d_inner_grid_ends, int *d_inner_cell_points,
                                                    int *d_inner_cell_dim_ids,
                                                    float *d_D_current, float *d_sum_sin, float *d_sum_cos,
                                                    int outer_grid_width, int outer_grid_dims, float outer_cell_size,
                                                    int inner_grid_width, float inner_cell_size,
                                                    int n, int d, float eps, float eps_extra) {


    int outer_grid_radius = ceil(eps_extra / outer_cell_size);

    extern __shared__ float s_d[];

    float *mbr = &s_d[threadIdx.x * (d + 4 * outer_grid_dims)];
    int *tmp = (int *) &mbr[d];
    int *idxs = &tmp[outer_grid_dims];

    int *tmp_extra = &idxs[outer_grid_dims];
    int *idxs_extra = &tmp_extra[outer_grid_dims];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < d_to_be_checked_size[0]; i += blockDim.x * gridDim.x) {

        if (d_r_local[0] == 0.) {
            return;
        }

        int p = d_to_be_checked[i * 2 + 0];
        int q = d_to_be_checked[i * 2 + 1];

        if (MBR_in_eps(mbr, tmp_extra, idxs_extra, d_outer_grid_ends,
                       d_inner_grid_ends, d_inner_cell_points, d_inner_cell_dim_ids,
                       d_D_current,
                       outer_grid_width, outer_grid_dims, outer_cell_size,
                       inner_cell_size,
                       d, eps, p, q)) {
            d_r_local[0] = 0.;

            return;
        }
    }
}

__global__
void kernel_reorder_data(float *d_D_current, float *d_D_next, int *d_inner_grid_points, int n, int d) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        int p_idx = d_inner_grid_points[i];
        for (int j = 0; j < d; j++) {
            d_D_current[i * d + j] = d_D_next[p_idx * d + j];
        }
    }
}

__global__
void kernel_check_dataset(int *check, float *d_D, int n, int d) {

    for (int p = threadIdx.x + blockIdx.x * blockDim.x; p < n; p += blockDim.x * gridDim.x) {
        for (int i = 0; i < d; i++) {
            float val = d_D[p * d + i];
            if (val < 0 - 0.000001 || val > 1 + 0.000001) {
                printf("val: %f\n", val);
                check[0] = 1;
                return;
            }
        }
    }
}


/// changing the spaceuse of the grid structure
/// im trying to have a course grained lvl and then a list of cells at the correct size
clustering
GPU_DynamicalClustering_DOUBLE_GRID(float *h_D, int n, int d, float eps, float lam, int cell_factor, int version) {
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

//    int *check = gpu_malloc_int(1);
//    cudaMemset(check, 0, sizeof(int));
//    kernel_check_dataset<<<number_of_blocks, min(n, BLOCK_SIZE)>>>(check, d_D_current, n, d);
//    cudaMemset(check, 0, sizeof(int));
//    kernel_check_dataset<<<number_of_blocks, min(n, BLOCK_SIZE)>>>(check, d_D_next, n, d);

    //computing the sizes for the inner grid
    float inner_cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int inner_grid_width = 1 / inner_cell_size;
    int inner_grid_dims = d;

    //computing the sizes for the outer grid
    float outer_cell_size = inner_cell_size * cell_factor; // each outer cell contains the inner cells fully
    int outer_grid_width = 1 / outer_cell_size;
    int outer_grid_dims = min(d, (int) (log(n) / log(outer_grid_width)));
    int outer_grid_number_of_cells = pow(outer_grid_width, outer_grid_dims);

    int outer_grid_radius = ceil(eps / outer_cell_size);
    int outer_grid_neighborhood_width = 2 * outer_grid_radius + 1;
//
//    printf("outer_cell_size: %f, outer_grid_width: %d, outer_grid_dims: %d, outer_grid_number_of_cells: %d\n",
//           outer_cell_size, outer_grid_width, outer_grid_dims, outer_grid_number_of_cells);

    //allocating space for the outer grid
    int *d_outer_grid_sizes = gpu_malloc_int(outer_grid_number_of_cells);
    int *d_outer_grid_ends = gpu_malloc_int(outer_grid_number_of_cells);
    int *d_new_outer_grid_ends = gpu_malloc_int(outer_grid_number_of_cells);

    //allocating space for the inner grid
    int *d_inner_grid_sizes = gpu_malloc_int(n);
    int *d_inner_grid_ends = gpu_malloc_int(n);
    int *d_new_inner_grid_ends = gpu_malloc_int(n);
    int *d_inner_grid_included = gpu_malloc_int(n);
    int *d_inner_grid_idxs = gpu_malloc_int(n);
    int *d_inner_grid_points = gpu_malloc_int(n);
    int *d_inner_grid_cell_dim_ids = gpu_malloc_int(n * d);
    int *d_new_inner_grid_cell_dim_ids = gpu_malloc_int(n * d);


    //for MBR check
    int *d_to_be_checked_size = gpu_malloc_int(1);
    int *d_to_be_checked;// = gpu_malloc_int(2 * n);

    //for v=5
    int *d_pre_grid_non_empty;
    int *d_pre_grid_sizes;
    int *d_pre_grid_ends;
    int *d_pre_grid_cells;
    int *d_pre_inner_grid_sizes;
    int *d_pre_inner_grid_ends;
    int *d_pre_inner_grid_cells;
    int *d_pre_inner_grid_fully_sizes;
    int *d_pre_inner_grid_fully_ends;
    int *d_pre_inner_grid_fully_cells;
    if (version == 5) {
        int possible_neighbors = outer_grid_number_of_cells * pow(outer_grid_neighborhood_width, outer_grid_dims);
        d_pre_grid_non_empty = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_sizes = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_ends = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_cells = gpu_malloc_int(possible_neighbors);
    }
    if (version == 6) {
        int possible_neighbors = outer_grid_number_of_cells * pow(outer_grid_neighborhood_width, outer_grid_dims);
        d_pre_grid_non_empty = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_sizes = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_ends = gpu_malloc_int(outer_grid_number_of_cells);
        d_pre_grid_cells = gpu_malloc_int(possible_neighbors);

        possible_neighbors = n * pow(outer_grid_neighborhood_width, d);
        d_pre_inner_grid_sizes = gpu_malloc_int(n);
        d_pre_inner_grid_ends = gpu_malloc_int(n);
        d_pre_inner_grid_cells = gpu_malloc_int(possible_neighbors);
        d_pre_inner_grid_fully_sizes = gpu_malloc_int(n);
        d_pre_inner_grid_fully_ends = gpu_malloc_int(n);
        d_pre_inner_grid_fully_cells = gpu_malloc_int(possible_neighbors);
    }

    //sum of sin and cos for each inner grid cell
    float *d_sum_cos = gpu_malloc_float(n * d);
    float *d_sum_sin = gpu_malloc_float(n * d);

    //alocating for final clustering
    clustering C;
    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

    build_the_grid(d_outer_grid_sizes, d_outer_grid_ends, d_new_outer_grid_ends,
                   d_inner_grid_sizes, d_inner_grid_ends, d_new_inner_grid_ends,
                   d_inner_grid_included, d_inner_grid_idxs,
                   d_inner_grid_points, d_inner_grid_cell_dim_ids, d_new_inner_grid_cell_dim_ids,
                   d_D_current, d_sum_sin, d_sum_cos,
                   n, d,
                   outer_grid_number_of_cells, outer_grid_width, outer_grid_dims, outer_cell_size,
                   inner_grid_width, inner_grid_dims, inner_cell_size);

    if (version == 7) {
        kernel_reorder_data << < number_of_blocks, min(n, BLOCK_SIZE) >> >
                                                   (d_D_current, d_D_next, d_inner_grid_points, n, d);
    }

    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;

    while (r_local < lam) {//&& itr < 100) {
        itr++;


        gpu_set_all(d_r_local, 1, 1.);

        build_the_grid(d_outer_grid_sizes, d_outer_grid_ends, d_new_outer_grid_ends,
                       d_inner_grid_sizes, d_inner_grid_ends, d_new_inner_grid_ends,
                       d_inner_grid_included, d_inner_grid_idxs,
                       d_inner_grid_points, d_inner_grid_cell_dim_ids, d_new_inner_grid_cell_dim_ids,
                       d_D_current, d_sum_sin, d_sum_cos,
                       n, d,
                       outer_grid_number_of_cells, outer_grid_width, outer_grid_dims, outer_cell_size,
                       inner_grid_width, inner_grid_dims, inner_cell_size);


//        if(itr>30) {
//            int *h_outer_grid_sizes = copy_D_to_H(d_outer_grid_sizes, outer_grid_width * outer_grid_width);
//            print_array(h_outer_grid_sizes, outer_grid_width, outer_grid_width);
//            delete h_outer_grid_sizes;
//        }

        gpu_set_all_zero(d_sum_cos, n * d);
        gpu_set_all_zero(d_sum_sin, n * d);

        kernel_inner_grid_stats << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_sum_cos, d_sum_sin,
                d_outer_grid_ends,
                d_inner_grid_ends, d_inner_grid_cell_dim_ids,
                d_D_current,
                outer_grid_width, outer_grid_dims,
                outer_cell_size,
                inner_grid_width, inner_cell_size,
                n, d);

        /// do one iteration
        if (version == 2) {
            kernel_itr_grid_STAD5_2 << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local,
                    d_outer_grid_ends,
                    d_inner_grid_ends, d_inner_grid_points,
                    d_inner_grid_cell_dim_ids,
                    d_D_next, d_D_current, d_sum_sin,
                    d_sum_cos,
                    outer_grid_width, outer_grid_dims,
                    outer_cell_size,
                    inner_grid_width, inner_cell_size,
                    n, d, eps);
            gpuErrchk(cudaPeekAtLastError());
        } else if (version == 3) {
            kernel_itr_grid_STAD5_3 << < min(1000, n), 32 >> > (d_r_local,
                    d_outer_grid_ends,
                    d_inner_grid_ends, d_inner_grid_points,
                    d_inner_grid_cell_dim_ids,
                    d_D_next, d_D_current, d_sum_sin, d_sum_cos,
                    outer_grid_width, outer_grid_dims, outer_cell_size,
                    inner_grid_width, inner_cell_size,
                    n, d, eps);
            gpuErrchk(cudaPeekAtLastError());
        } else if (version == 4) {

            kernel_itr_grid_STAD5_4 << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_r_local,
                    d_outer_grid_ends,
                    d_inner_grid_ends, d_inner_grid_points,
                    d_inner_grid_cell_dim_ids,
                    d_D_next, d_D_current, d_sum_sin,
                    d_sum_cos,
                    outer_grid_width, outer_grid_dims,
                    outer_cell_size,
                    inner_grid_width, inner_cell_size,
                    n, d, eps);
        } else if (version == 5) {
            int number_of_blocks_pre = outer_grid_number_of_cells / BLOCK_SIZE;
            if (outer_grid_number_of_cells % BLOCK_SIZE) number_of_blocks_pre++;

            gpu_set_all_zero(d_pre_grid_sizes, 1);

            kernel_itr_grid_STAD5_5non_empty << < number_of_blocks_pre, min(outer_grid_number_of_cells, BLOCK_SIZE) >> >
                                                                        (d_pre_grid_non_empty, d_pre_grid_sizes, d_outer_grid_ends,
                                                                                outer_grid_number_of_cells);

            int non_empty_cells = copy_last_D_to_H(d_pre_grid_sizes, 1);


            int number_of_blocks_ne = non_empty_cells / BLOCK_SIZE;
            if (non_empty_cells % BLOCK_SIZE) number_of_blocks_ne++;

            gpu_set_all_zero(d_pre_grid_sizes, outer_grid_number_of_cells);
            gpu_set_all_zero(d_pre_grid_ends, outer_grid_number_of_cells);


            kernel_itr_grid_STAD5_5pre_size_1 << < number_of_blocks_ne, min(non_empty_cells,
                                                                            BLOCK_SIZE) >> >
                                                                        (d_pre_grid_non_empty, d_pre_grid_sizes, d_outer_grid_ends,
                                                                                non_empty_cells, outer_grid_dims, outer_grid_width,
                                                                                eps, outer_cell_size);

            inclusive_scan(d_pre_grid_sizes, d_pre_grid_ends, outer_grid_number_of_cells);

            gpu_set_all_zero(d_pre_grid_sizes, outer_grid_number_of_cells);

//            int possible_neighbors = outer_grid_number_of_cells * pow(outer_grid_neighborhood_width, outer_grid_dims);

            kernel_itr_grid_STAD5_5pre_populate_1 << < number_of_blocks_ne,
                    min(non_empty_cells, BLOCK_SIZE) >> >
                    (d_pre_grid_non_empty, d_pre_grid_cells, d_pre_grid_ends, d_pre_grid_sizes, d_outer_grid_ends,
                            non_empty_cells, outer_grid_dims, outer_grid_width,
                            eps, outer_cell_size);

//            cudaMemset(check, 0, sizeof(int));
            kernel_itr_grid_STAD5_5_1 << < number_of_blocks, min(n, BLOCK_SIZE), BLOCK_SIZE * d * 2 * sizeof(float) >> >
                                                                                 (d_pre_grid_cells, d_pre_grid_ends, d_r_local,
                                                                                         d_outer_grid_ends,
                                                                                         d_inner_grid_ends, d_inner_grid_points,
                                                                                         d_inner_grid_cell_dim_ids,
                                                                                         d_D_next, d_D_current, d_sum_sin, d_sum_cos,
                                                                                         outer_grid_width, outer_grid_dims,
                                                                                         outer_cell_size,
                                                                                         inner_grid_width, inner_cell_size,
                                                                                         n, d, eps, itr);
        } else if (version == 6) {
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());


            int number_of_blocks_pre = outer_grid_number_of_cells / BLOCK_SIZE;
            if (outer_grid_number_of_cells % BLOCK_SIZE) number_of_blocks_pre++;

            gpu_set_all_zero(d_pre_grid_sizes, 1);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            kernel_itr_grid_STAD5_5non_empty << < number_of_blocks_pre, min(outer_grid_number_of_cells, BLOCK_SIZE) >> >
                                                                        (
                                                                                d_pre_grid_non_empty, d_pre_grid_sizes, d_outer_grid_ends,
                                                                                        outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            int non_empty_cells = copy_last_D_to_H(d_pre_grid_sizes, 1);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            int number_of_blocks_ne = non_empty_cells / BLOCK_SIZE;
            if (non_empty_cells % BLOCK_SIZE) number_of_blocks_ne++;

            gpu_set_all_zero(d_pre_grid_sizes, outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            gpu_set_all_zero(d_pre_grid_ends, outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            kernel_itr_grid_STAD5_5pre_size_1 << < number_of_blocks_ne, min(non_empty_cells,
                                                                            BLOCK_SIZE) >> >
                                                                        (d_pre_grid_non_empty, d_pre_grid_sizes, d_outer_grid_ends,
                                                                                non_empty_cells, outer_grid_dims, outer_grid_width,
                                                                                eps, outer_cell_size);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            inclusive_scan(d_pre_grid_sizes, d_pre_grid_ends, outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            gpu_set_all_zero(d_pre_grid_sizes, outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            kernel_itr_grid_STAD5_5pre_populate_1 << < number_of_blocks_ne,
                    min(non_empty_cells, BLOCK_SIZE) >> >
                    (d_pre_grid_non_empty, d_pre_grid_cells, d_pre_grid_ends, d_pre_grid_sizes, d_outer_grid_ends,
                            non_empty_cells, outer_grid_dims, outer_grid_width,
                            eps, outer_cell_size);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());


            int inner_grid_number_of_cells = copy_last_D_to_H(d_outer_grid_ends, outer_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            printf("inner_grid_number_of_cells: %d\n", inner_grid_number_of_cells);
            int number_of_blocks_pre_inner = inner_grid_number_of_cells / BLOCK_SIZE;
            if (inner_grid_number_of_cells % BLOCK_SIZE) number_of_blocks_pre_inner++;

            gpu_set_all_zero(d_pre_inner_grid_sizes, inner_grid_number_of_cells);
            gpu_set_all_zero(d_pre_inner_grid_fully_sizes, inner_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            gpu_set_all_zero(d_pre_inner_grid_ends, inner_grid_number_of_cells);
            gpu_set_all_zero(d_pre_inner_grid_fully_ends, inner_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

//            kernel_itr_grid_STAD5_6pre_size << < number_of_blocks_pre_inner, min(inner_grid_number_of_cells,
//                                                                                 BLOCK_SIZE) >> >
//            (d_pre_grid_ends, d_pre_grid_cells, d_pre_inner_grid_sizes, d_pre_inner_grid_fully_sizes, d_outer_grid_ends,
//                    d_inner_grid_ends, d_inner_grid_cell_dim_ids,
//                    inner_grid_number_of_cells, outer_grid_dims, outer_grid_width, d,
//                    eps, outer_cell_size);
//            cudaDeviceSynchronize();
//            gpuErrchk(cudaPeekAtLastError());

            kernel_itr_grid_STAD5_6_1pre_size << < number_of_blocks_pre_inner, min(inner_grid_number_of_cells,
                                                                                   BLOCK_SIZE) >> >
                                                                               (d_pre_grid_ends, d_pre_grid_cells, d_pre_inner_grid_sizes, d_pre_inner_grid_fully_sizes, d_outer_grid_ends,
                                                                                       d_inner_grid_ends, d_inner_grid_cell_dim_ids,
                                                                                       inner_grid_number_of_cells, outer_grid_dims, outer_grid_width, d,
                                                                                       eps, outer_cell_size);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            inclusive_scan(d_pre_inner_grid_sizes, d_pre_inner_grid_ends, inner_grid_number_of_cells);
            inclusive_scan(d_pre_inner_grid_fully_sizes, d_pre_inner_grid_fully_ends, inner_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            gpu_set_all_zero(d_pre_inner_grid_sizes, inner_grid_number_of_cells);
            gpu_set_all_zero(d_pre_inner_grid_fully_sizes, inner_grid_number_of_cells);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

//            kernel_itr_grid_STAD5_6pre_populate << < number_of_blocks_pre_inner,
//                    min(inner_grid_number_of_cells, BLOCK_SIZE) >> >
//            (d_pre_grid_ends, d_pre_grid_cells,
//                    d_pre_inner_grid_cells, d_pre_inner_grid_ends,
//                    d_pre_inner_grid_sizes,
//                    d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
//                    d_pre_inner_grid_fully_sizes,
//                    d_outer_grid_ends,
//                    d_inner_grid_ends, d_inner_grid_cell_dim_ids,
//                    inner_grid_number_of_cells, outer_grid_dims, outer_grid_width, d,
//                    eps, outer_cell_size);

//            printf("d_inner_grid_ends: \n");
//            print_array_gpu(d_inner_grid_ends, inner_grid_number_of_cells);

            kernel_itr_grid_STAD5_6_1pre_populate << < number_of_blocks_pre_inner,
                    min(inner_grid_number_of_cells, BLOCK_SIZE) >> >
                    (d_pre_grid_ends, d_pre_grid_cells,
                            d_pre_inner_grid_cells, d_pre_inner_grid_ends,
                            d_pre_inner_grid_sizes,
                            d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
                            d_pre_inner_grid_fully_sizes,
                            d_outer_grid_ends,
                            d_inner_grid_points, d_inner_grid_ends, d_inner_grid_cell_dim_ids,
                            inner_grid_number_of_cells, outer_grid_dims, outer_grid_width, d,
                            eps, outer_cell_size);
//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

//            kernel_itr_grid_STAD5_6 << < number_of_blocks, min(n, BLOCK_SIZE) >> >
//            (d_pre_inner_grid_cells, d_pre_inner_grid_ends,
//                    d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
//                    d_r_local,
//                    d_outer_grid_ends,
//                    d_inner_grid_ends, d_inner_grid_points,
//                    d_inner_grid_cell_dim_ids,
//                    d_D_next, d_D_current, d_sum_sin, d_sum_cos,
//                    outer_grid_width, outer_grid_dims,
//                    outer_cell_size,
//                    inner_grid_width, inner_cell_size,
//                    n, d, eps);

            kernel_itr_grid_STAD5_6_1 << < number_of_blocks, min(n, BLOCK_SIZE), BLOCK_SIZE * d * sizeof(float) >> >
                                                                                 (d_pre_inner_grid_cells, d_pre_inner_grid_ends,
                                                                                         d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
                                                                                         d_r_local,
                                                                                         d_outer_grid_ends,
                                                                                         d_inner_grid_ends, d_inner_grid_points,
                                                                                         d_inner_grid_cell_dim_ids,
                                                                                         d_D_next, d_D_current, d_sum_sin, d_sum_cos,
                                                                                         outer_grid_width, outer_grid_dims,
                                                                                         outer_cell_size,
                                                                                         inner_grid_width, inner_cell_size,
                                                                                         n, d, eps);

//            kernel_itr_grid_STAD5_6_2 << < inner_grid_number_of_cells, min(n, 64)>> >
//            (d_pre_inner_grid_cells, d_pre_inner_grid_ends,
//                    d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
//                    d_r_local,
//                    d_outer_grid_ends,
//                    d_inner_grid_ends, d_inner_grid_points,
//                    d_inner_grid_cell_dim_ids,
//                    d_D_next, d_D_current, d_sum_sin, d_sum_cos,
//                    outer_grid_width, outer_grid_dims,
//                    outer_cell_size,
//                    inner_grid_width, inner_cell_size,
//                    n, d, eps);

//            kernel_itr_grid_STAD5_6_3 << < inner_grid_number_of_cells, 64, 3 * 64 * d * sizeof(float)>> >
//            (d_pre_inner_grid_cells, d_pre_inner_grid_ends,
//                    d_pre_inner_grid_fully_cells, d_pre_inner_grid_fully_ends,
//                    d_r_local,
//                    d_outer_grid_ends,
//                    d_inner_grid_ends, d_inner_grid_points,
//                    d_inner_grid_cell_dim_ids,
//                    d_D_next, d_D_current, d_sum_sin, d_sum_cos,
//                    outer_grid_width, outer_grid_dims,
//                    outer_cell_size,
//                    inner_grid_width, inner_cell_size,
//                    n, d, eps);

//            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
        } else {

            kernel_itr_grid_STAD5 << < number_of_blocks, min(n, BLOCK_SIZE),
                    BLOCK_SIZE * (d + 2 * outer_grid_dims) * sizeof(float) >> > (d_r_local,
                            d_outer_grid_ends,
                            d_inner_grid_ends, d_inner_grid_points,
                            d_inner_grid_cell_dim_ids,
                            d_D_next, d_D_current, d_sum_sin, d_sum_cos,
                            outer_grid_width, outer_grid_dims,
                            outer_cell_size,
                            inner_grid_width, inner_cell_size,
                            n, d, eps, itr);
        }

        r_local = copy_last_D_to_H(d_r_local, 1);

        printf("itr: %d, r_local: %f\n", itr, r_local);


//        if (itr > 200) {
//            throw std::exception();
//        }

        if (version != 0 && r_local >= lam) {

            float eps_extra = 2 * eps - eps * sqrt(15. / 16.) + eps / 2. - sin(eps / 2.);

//            kernel_float_grid_extra_check << < number_of_blocks, min(n, BLOCK_SIZE),
//                    BLOCK_SIZE * (d + 4 * outer_grid_dims) * sizeof(float)>> > (d_r_local,
//                    d_outer_grid_ends,
//                    d_inner_grid_ends,
//                    d_inner_grid_points,
//                    d_inner_grid_cell_dim_ids,
//                    d_D_current, d_sum_sin, d_sum_cos,
//                    outer_grid_width, outer_grid_dims,
//                    outer_cell_size,
//                    inner_grid_width, inner_cell_size,
//                    n, d, eps, eps_extra);
//            cudaDeviceSynchronize();
//            gpuErrchk(cudaPeekAtLastError());

            cudaMemset(d_to_be_checked_size, 0, sizeof(int));
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            kernel_float_grid_extra_check_part1_count << < number_of_blocks, min(n, BLOCK_SIZE),
                    BLOCK_SIZE * (d + 2 * outer_grid_dims) * sizeof(float) >> > (d_to_be_checked_size, d_to_be_checked,
                            d_r_local,
                            d_outer_grid_ends,
                            d_inner_grid_ends,
                            d_inner_grid_points,
                            d_inner_grid_cell_dim_ids,
                            d_D_current, d_sum_sin, d_sum_cos,
                            outer_grid_width, outer_grid_dims,
                            outer_cell_size,
                            inner_grid_width, inner_cell_size,
                            n, d, eps, eps_extra);

            int size = copy_last_D_to_H(d_to_be_checked_size, 1);
            d_to_be_checked = gpu_malloc_int(2 * size);
//            printf("size: %d\n", size);
            cudaMemset(d_to_be_checked_size, 0, sizeof(int));

            kernel_float_grid_extra_check_part1 << < number_of_blocks, min(n, BLOCK_SIZE),
                    BLOCK_SIZE * (d + 2 * outer_grid_dims) * sizeof(float) >> > (d_to_be_checked_size, d_to_be_checked,
                            d_r_local,
                            d_outer_grid_ends,
                            d_inner_grid_ends,
                            d_inner_grid_points,
                            d_inner_grid_cell_dim_ids,
                            d_D_current, d_sum_sin, d_sum_cos,
                            outer_grid_width, outer_grid_dims,
                            outer_cell_size,
                            inner_grid_width, inner_cell_size,
                            n, d, eps, eps_extra);


            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());

            kernel_float_grid_extra_check_part2 << < number_of_blocks, min(n, BLOCK_SIZE),
                    BLOCK_SIZE * (d + 4 * outer_grid_dims) * sizeof(float) >> > (d_to_be_checked_size, d_to_be_checked,
                            d_r_local,
                            d_outer_grid_ends,
                            d_inner_grid_ends,
                            d_inner_grid_points,
                            d_inner_grid_cell_dim_ids,
                            d_D_current, d_sum_sin, d_sum_cos,
                            outer_grid_width, outer_grid_dims,
                            outer_cell_size,
                            inner_grid_width, inner_cell_size,
                            n, d, eps, eps_extra);

            r_local = copy_last_D_to_H(d_r_local, 1);
        }

        swap(d_D_current, d_D_next);
    }
    if (itr == 100) {
        printf("kept running!\n");
    }

    build_the_grid(d_outer_grid_sizes, d_outer_grid_ends, d_new_outer_grid_ends,
                   d_inner_grid_sizes, d_inner_grid_ends, d_new_inner_grid_ends,
                   d_inner_grid_included, d_inner_grid_idxs,
                   d_inner_grid_points, d_inner_grid_cell_dim_ids, d_new_inner_grid_cell_dim_ids,
                   d_D_current, d_sum_sin, d_sum_cos,
                   n, d,
                   outer_grid_number_of_cells, outer_grid_width, outer_grid_dims, outer_cell_size,
                   inner_grid_width, inner_grid_dims, inner_cell_size);

    GPU_synCluster_float_grid << < number_of_blocks, min(n, BLOCK_SIZE) >> > (d_C, d_D_current,
            d_outer_grid_ends,
            d_inner_grid_cell_dim_ids, d_inner_grid_ends,
            outer_grid_width, outer_grid_dims,
            outer_cell_size,
            inner_grid_width, inner_cell_size,
            n, d);

    int *h_C = copy_D_to_H(d_C, n);

    int k = maximum(h_C, n) + 1;
    if (k > 0) {
        for (int i = 0; i < k; i++) {
            cluster c;
            C.push_back(c);
        }
        for (int p = 0; p < n; p++) {
            if (h_C[p] >= 0)
                C[h_C[p]].push_back(p);
        }
    }

    if (version == 5) {
        cudaFree(d_pre_grid_sizes);
        cudaFree(d_pre_grid_ends);
        cudaFree(d_pre_grid_cells);
        cudaFree(d_pre_grid_non_empty);
    }

    if (version == 6) {
        cudaFree(d_pre_grid_sizes);
        cudaFree(d_pre_grid_ends);
        cudaFree(d_pre_grid_cells);
        cudaFree(d_pre_grid_non_empty);
        cudaFree(d_pre_inner_grid_sizes);
        cudaFree(d_pre_inner_grid_ends);
        cudaFree(d_pre_inner_grid_cells);
        cudaFree(d_pre_inner_grid_fully_cells);
        cudaFree(d_pre_inner_grid_fully_ends);
        cudaFree(d_pre_inner_grid_fully_sizes);
    }

    //delete grid structure
    cudaFree(d_outer_grid_sizes);
    cudaFree(d_outer_grid_ends);
    cudaFree(d_new_outer_grid_ends);
    cudaFree(d_inner_grid_sizes);
    cudaFree(d_inner_grid_ends);
    cudaFree(d_new_inner_grid_ends);
    cudaFree(d_inner_grid_included);
    cudaFree(d_inner_grid_idxs);
    cudaFree(d_inner_grid_points);
    cudaFree(d_inner_grid_cell_dim_ids);
    cudaFree(d_new_inner_grid_cell_dim_ids);

    //delete temp data
    cudaFree(d_D_current);
    cudaFree(d_D_next);

    //delete GPU result
    cudaFree(d_C);
    cudaFree(d_incl);
    cudaFree(d_map);

    //delete summarization
    cudaFree(d_sum_cos);
    cudaFree(d_sum_sin);

    //delete variables
    cudaFree(d_r_local);

    delete h_C;
    gpuErrchk(cudaPeekAtLastError());

    return C;
}