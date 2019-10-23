#pragma once

#include <cmath>

#include "util/primitives/primitives.h"
#include "util/timer.h"
#include "util/util.h"
#include "parsing_util.h"

#include "cuda/cuda_base.cuh"
#include "cuda/CUDAStat.cuh"

template<typename T>
T *GetMMAPArr(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    auto ret = ftruncate(file_fd, file_size);

    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_WRITE, MAP_SHARED, file_fd, 0);
//    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_WRITE, MAP_SHARED | MAP_POPULATE, file_fd, 0);
    log_info("Open & MMAP Time: %.6lfs", populate_timer.elapsed());
    assert(ret == 0);
    return mmap_arr_;
}

template<typename T>
T *GetMMAPArrReadOnly(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    assert(file_fd >= 0);

    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);
    log_info("Open & MMAP Time: %.6lfs", populate_timer.elapsed());
    return mmap_arr_;
}

template<typename T>
T *GetMallocPReadArrReadOnly(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    log_info("File Size: %zu", file_size);
    assert(file_fd >= 0);

#ifdef USE_GPU
    char *arr = nullptr;
    CUDA_MALLOC(&arr, sizeof(char)*file_size, nullptr);
#else
    auto arr = (char *) malloc(file_size);
#endif

#pragma omp parallel for
    for (size_t i = 0; i < file_size; i += IO_REQ_SIZE) {
        auto size = min(i + IO_REQ_SIZE, file_size) - i;
//        assert(size <= IO_REQ_SIZE);
        auto ret = pread(file_fd, arr + i, size, i);
//        assert(ret == size);
    }
    log_info("Open & Malloc & PRead Time: %.6lfs", populate_timer.elapsed());
    return (T *) arr;
}

/*
 * Global One.
 */
class FileLoader {
protected:
    int file_fd{};
    Timer timer;
public:
    size_t size{};

    FileLoader() = default;

    explicit FileLoader(const char *file_name) {
        file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
        size = file_size(file_name);
        log_info("Start IO");
    }

    ssize_t ReadToBuf(size_t i, char *tmp) {
        ssize_t num_reads;
        if (i != 0) {
            num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
        } else {
            tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
            num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i)
                        + EXTRA_IO_SIZE;
        }
        return num_reads;
    }

    void PrintEndStat() {
        log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
    }
};

class FileLoaderMMap : public FileLoader {
    char *mmap_mem;
public:
    explicit FileLoaderMMap(const char *file_name) : FileLoader(file_name) {
        mmap_mem = (char *) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_fd, 0);
    }

    ssize_t ReadToBuf(size_t i, char *&tmp, char *first_buf) {
        ssize_t num_reads;
        if (i != 0) {
            num_reads = min<ssize_t>(size - i + EXTRA_IO_SIZE, IO_REQ_SIZE);
            tmp = mmap_mem + i - EXTRA_IO_SIZE;
        } else {
            num_reads = min<ssize_t>(size - i, IO_REQ_SIZE - EXTRA_IO_SIZE);
            memcpy(first_buf + EXTRA_IO_SIZE, mmap_mem, num_reads);
            tmp = first_buf;
            tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
            num_reads += EXTRA_IO_SIZE;
        }
        return num_reads;
    }
};