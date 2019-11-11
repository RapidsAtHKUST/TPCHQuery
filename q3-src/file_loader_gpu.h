#pragma once

#include "util/timer.h"
#include "util/util.h"
#include "parsing_util.h"

#include "cuda/cuda_base.cuh"
#include "cuda/CUDAStat.cuh"

template<typename T>
T *GetMallocPReadArrReadOnlyGPU(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    log_info("File Size: %zu", file_size);
    assert(file_fd >= 0);

    char *arr = nullptr;
#ifdef UM
    CUDA_MALLOC(&arr, sizeof(char)*file_size, nullptr);
#else
    char *d_arr = nullptr;
    log_trace("Allocate %lu bytes.", sizeof(char)*file_size);
    checkCudaErrors(cudaMalloc((void**)&d_arr, sizeof(char)*file_size));
    arr = (char *) malloc(file_size);
#endif
    log_info("After malloc: %.2f s.", populate_timer.elapsed());

#pragma omp parallel for
    for (size_t i = 0; i < file_size; i += IO_REQ_SIZE) {
        auto size = min(i + IO_REQ_SIZE, file_size) - i;
//        assert(size <= IO_REQ_SIZE);
        auto ret = pread(file_fd, arr + i, size, i);
//        assert(ret == size);
    }
    log_info("Open & Malloc & PRead Time: %.6lfs", populate_timer.elapsed());
#ifdef UM
    return (T *) arr;
#else
    checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(char)*file_size, cudaMemcpyHostToDevice));
    free(arr);
    return (T *) d_arr;
#endif
}