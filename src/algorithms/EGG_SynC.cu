//
// Created by jakobrj on 4/22/22.
//
#include "EGG_SynC.cuh"
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
            int g = d_inner_cell_dim_ids[inner_cell_idx * d + i];
            float val = d_D_current[p * d + i];
//            atomicAdd(&d_sum_cos[inner_cell_idx * d + i], cos(val)); //- cos(l * inner_cell_size));
//            atomicAdd(&d_sum_sin[inner_cell_idx * d + i], sin(val)); //- sin(l * inner_cell_size));
            atomicAdd(&d_sum_cos[inner_cell_idx * d + i], cos(val) - cos(g * inner_cell_size));
            atomicAdd(&d_sum_sin[inner_cell_idx * d + i], sin(val) - sin(g * inner_cell_size));
        }
    }
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


__device__ float gpu_distance(int p, int q, const float *d_D, int d) {

    float dist = 0.;
    for (int l = 0; l < d; l++) {
        float diff = d_D[p * d + l] - d_D[q * d + l];
        dist += diff * diff;
    }
    dist = sqrt(dist);

    return dist;
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
    float *sum = &s_d[threadIdx.x * d * 3];
    float *sum_small = &sum[d];
    float *x = &sum_small[d];

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
            sum_small[l] = 0.;
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
//                fully_included = false;//todo remove
                if (fully_included) {
                    for (int l = 0; l < d; l++) {

//                        sum[l] += ((d_sum_sin[inner_cell_idx * d + l] * cos(x[l])) -
//                                   (d_sum_cos[inner_cell_idx * d + l] * sin(x[l])));

                        int g = d_inner_cell_dim_ids[inner_cell_idx * d + l];

                        const float x_t = d_D_current[p * d + l];
                        const float to_be_added = (((sin(g * inner_cell_size) * inner_cell_number_of_points +
                                                     d_sum_sin[inner_cell_idx * d + l]) * cos(x_t)) -
                                                   ((cos(g * inner_cell_size) * inner_cell_number_of_points +
                                                     d_sum_cos[inner_cell_idx * d + l]) * sin(x_t)));

//                        if (p == 87378) {
//                            printf("c: %d, to_be_added[%d]: %f\n", inner_cell_idx, l, to_be_added);
//                        }

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
//                                    for (int l = 0; l < d; l++) {
//                                        sum_small[l] += sin(d_D_current[q * d + l] - x[l]);
//                                    }
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
//                            if(itr >= 30 && p==95860){
//                                printf("q: %d\n", q);
//                            }
                            for (int l = 0; l < d; l++) {
                                sum_small[l] += sin(d_D_current[q * d + l] - x[l]);
                            }
                            number_of_neighbors++;
                        }
                    }
                }
            }
        }

        for (int l = 0; l < d; l++) {
            d_D_next[p * d + l] = x[l] + (sum[l] + sum_small[l]) / number_of_neighbors;
        }


        if ((!on_boarder && number_of_neighbors != center_inner_cell_number_of_points) ||
            (on_boarder && number_of_neighbors != half_count)) {
            d_r_local[0] = 0.;

            if (itr > 30 && center_inner_cell_number_of_points == 1) {
//                printf("itr: %d, p: %d, c: %d full: %d, half: %d, lower bound half: %d, on_boarder: %d\n", itr, p,
//                       center_inner_cell_idx,
//                       number_of_neighbors, half_count,
//                       center_inner_cell_number_of_points, on_boarder ? 1 : 0);
//
//                printf("update: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.10f ", (sum[l] + sum_small[l]) / number_of_neighbors);
//                }
//                printf("\n");

//                printf("location: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.20f ", d_D_next[p * d + l]);
//                }
//                printf("\n");

//
//                printf("sum: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.10f ", sum[l]);
//                }
//                printf("\n");
//
//                printf("sum_small: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.10f ", sum_small[l]);
//                }
//                printf("\n");
            }
//
//            if (itr > 30 && 80 >= center_inner_cell_number_of_points && 70 <= center_inner_cell_number_of_points &&
//                p == 87217) {
//                printf("itr: %d, p: %d, c: %d full: %d, half: %d, lower bound half: %d, on_boarder: %d\n", itr, p,
//                       center_inner_cell_idx,
//                       number_of_neighbors, half_count,
//                       center_inner_cell_number_of_points, on_boarder ? 1 : 0);
//
//                printf("update: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.10f ", (sum[l] + sum_small[l]) / number_of_neighbors);
//                }
//                printf("\n");
//
//                printf("location: ");
//                for (int l = 0; l < d; l++) {
//                    printf("%0.10f ", d_D_next[p * d + l]);
//                }
//                printf("\n");
//            }
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

                                    int i = atomicInc((unsigned int *) d_to_be_checked_size, n * n);//todo why i?

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

std::vector <at::Tensor> EGG_SynC(float *h_D, int n, int d, float eps) {

    gpu_reset_max_memory_usage();

    float lam = 0.001; //todo fix

    Timer<6> timer;

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.start_stage_time(0);

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE)
        number_of_blocks++;

    float *d_D_current = copy_H_to_D(h_D, n * d);
    float *d_D_next = copy_D_to_D(d_D_current, n * d);

    //computing the sizes for the inner grid
    float inner_cell_size = sqrt(pow((eps / 2.), 2.) / d);
    int inner_grid_width = 1 / inner_cell_size;
    int inner_grid_dims = d;

    //computing the sizes for the outer grid
    float outer_cell_size = inner_cell_size; // each outer cell contains the inner cells fully
    int outer_grid_width = 1 / outer_cell_size;
    int outer_grid_dims = min(d, (int) (log(n) / log(outer_grid_width)));
    int outer_grid_number_of_cells = pow(outer_grid_width, outer_grid_dims);

    int outer_grid_radius = ceil(eps / outer_cell_size);
    int outer_grid_neighborhood_width = 2 * outer_grid_radius + 1;

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

    int *d_pre_grid_non_empty;
    int *d_pre_grid_sizes;
    int *d_pre_grid_ends;
    int *d_pre_grid_cells;

    int possible_neighbors = outer_grid_number_of_cells * pow(outer_grid_neighborhood_width, outer_grid_dims);
    d_pre_grid_non_empty = gpu_malloc_int(outer_grid_number_of_cells);
    d_pre_grid_sizes = gpu_malloc_int(outer_grid_number_of_cells);
    d_pre_grid_ends = gpu_malloc_int(outer_grid_number_of_cells);
    d_pre_grid_cells = gpu_malloc_int(possible_neighbors);

    //sum of sin and cos for each inner grid cell
    float *d_sum_cos = gpu_malloc_float(n * d);
    float *d_sum_sin = gpu_malloc_float(n * d);

    //alocating for final clustering
    int *d_C = gpu_malloc_int(n);
    gpu_set_all(d_C, n, -1);
    int *d_incl = gpu_malloc_int_zero(n);
    int *d_map = gpu_malloc_int_zero(n);

//    build_the_grid(d_outer_grid_sizes, d_outer_grid_ends, d_new_outer_grid_ends,
//                   d_inner_grid_sizes, d_inner_grid_ends, d_new_inner_grid_ends,
//                   d_inner_grid_included, d_inner_grid_idxs,
//                   d_inner_grid_points, d_inner_grid_cell_dim_ids, d_new_inner_grid_cell_dim_ids,
//                   d_D_current, d_sum_sin, d_sum_cos,
//                   n, d,
//                   outer_grid_number_of_cells, outer_grid_width, outer_grid_dims, outer_cell_size,
//                   inner_grid_width, inner_grid_dims, inner_cell_size);


    float *d_r_local = gpu_malloc_float(1);
    float r_local = 0.;
    int itr = 0;


    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.end_stage_time(0);

    while (r_local < lam) {
        itr++;

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.start_itr_time();

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.start_stage_time(1);

        gpu_set_all(d_r_local, 1, 1.);

        build_the_grid(d_outer_grid_sizes, d_outer_grid_ends, d_new_outer_grid_ends,
                       d_inner_grid_sizes, d_inner_grid_ends, d_new_inner_grid_ends,
                       d_inner_grid_included, d_inner_grid_idxs,
                       d_inner_grid_points, d_inner_grid_cell_dim_ids, d_new_inner_grid_cell_dim_ids,
                       d_D_current, d_sum_sin, d_sum_cos,
                       n, d,
                       outer_grid_number_of_cells, outer_grid_width, outer_grid_dims, outer_cell_size,
                       inner_grid_width, inner_grid_dims, inner_cell_size);


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

        kernel_itr_grid_STAD5_5pre_populate_1 << < number_of_blocks_ne,
                min(non_empty_cells, BLOCK_SIZE) >> >
        (d_pre_grid_non_empty, d_pre_grid_cells, d_pre_grid_ends, d_pre_grid_sizes, d_outer_grid_ends,
                non_empty_cells, outer_grid_dims, outer_grid_width,
                eps, outer_cell_size);


        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.end_stage_time(1);
        timer.start_stage_time(2);

        kernel_itr_grid_STAD5_5_1 << < number_of_blocks, min(n, BLOCK_SIZE), BLOCK_SIZE * d * 3 * sizeof(float) >> >
        (d_pre_grid_cells, d_pre_grid_ends, d_r_local,
                d_outer_grid_ends,
                d_inner_grid_ends, d_inner_grid_points,
                d_inner_grid_cell_dim_ids,
                d_D_next, d_D_current, d_sum_sin, d_sum_cos,
                outer_grid_width, outer_grid_dims,
                outer_cell_size,
                inner_grid_width, inner_cell_size,
                n, d, eps, itr);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.end_stage_time(2);


        r_local = copy_last_D_to_H(d_r_local, 1);

        printf("itr: %d, r_local: %f\n", itr, r_local);


//        if (itr > 200) {
//            throw std::exception();
//        }

        if (r_local >= lam) { //todo should be just r_local not 1
            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            timer.start_stage_time(3);

            float eps_extra = 2 * eps - eps * sqrt(15. / 16.) + eps / 2. - sin(eps / 2.);

            cudaMemset(d_to_be_checked_size, 0, sizeof(int));
//            cudaDeviceSynchronize();
//            gpuErrchk(cudaPeekAtLastError());

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
            gpu_free(d_to_be_checked);//todo measure this part aswell

            cudaDeviceSynchronize();
            gpuErrchk(cudaPeekAtLastError());
            timer.end_stage_time(3);
        }

        swap(d_D_current, d_D_next);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
        timer.end_itr_time();
    }
    if (itr == 100) {
        printf("kept running!\n");
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.start_stage_time(4);
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
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    timer.end_stage_time(4);
    timer.start_stage_time(5);

    int *h_C = copy_D_to_H(d_C, n);

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

    gpu_free(d_to_be_checked_size);

    gpu_free(d_pre_grid_sizes);
    gpu_free(d_pre_grid_ends);
    gpu_free(d_pre_grid_cells);
    gpu_free(d_pre_grid_non_empty);

    //delete grid structure
    gpu_free(d_outer_grid_sizes);
    gpu_free(d_outer_grid_ends);
    gpu_free(d_new_outer_grid_ends);
    gpu_free(d_inner_grid_sizes);
    gpu_free(d_inner_grid_ends);
    gpu_free(d_new_inner_grid_ends);
    gpu_free(d_inner_grid_included);
    gpu_free(d_inner_grid_idxs);
    gpu_free(d_inner_grid_points);
    gpu_free(d_inner_grid_cell_dim_ids);
    gpu_free(d_new_inner_grid_cell_dim_ids);

    //delete temp data
    gpu_free(d_D_current);
    gpu_free(d_D_next);

    //delete GPU result
    gpu_free(d_C);
    gpu_free(d_incl);
    gpu_free(d_map);

    //delete summarization
    gpu_free(d_sum_cos);
    gpu_free(d_sum_sin);

    //delete variables
    gpu_free(d_r_local);

    if (gpu_total_memory_usage() != 0) {
        printf("\n\n\nNot all memory freed!!!\nMissing: %d\n\n\n", gpu_total_memory_usage());
    }

    delete h_C;
    gpuErrchk(cudaPeekAtLastError());


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
    timer.end_stage_time(5);

    return result;
}