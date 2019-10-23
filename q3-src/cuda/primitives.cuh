//
// Created by Bryan on 22/7/2019.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda/cuda_fake/fake.h"
#endif

#include "CUDAStat.cuh"
#include "../cuda/cub/cub/cub.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
using namespace std;

#define GRID_SIZE_DEFAULT   (1024)
#define BLOCK_SIZE_DEFAULT  (256)

/*CUDA kernels*/
template <typename DataType, typename IndexType, typename CntType>
__global__
void gather(DataType *input, DataType *output, IndexType *idxes, CntType cnt)
{
    CntType gtid = (CntType)(threadIdx.x + blockDim.x * blockIdx.x);
    CntType gnum = (CntType)(blockDim.x * gridDim.x);

    while (gtid < cnt)
    {
        output[gtid] = input[idxes[gtid]];
        gtid += gnum;
    }
}

/*wrapper of the CUB primitives*/
template<typename DataType, typename CntType>
DataType CUBMax(
        DataType *input,
        const CntType count,
        CUDAMemStat *memstat,
        CUDATimeStat *timer)
{
    DataType *maxVal = nullptr;
    CUDA_MALLOC(&maxVal, sizeof(DataType), memstat);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input, maxVal, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input, maxVal, count), timer);

    DataType res = *maxVal;
    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(maxVal, memstat);

    return res;
}

template<typename InputType, typename OutputType, typename CntType>
OutputType CUBScanExclusive(
        InputType *input,
        OutputType *output,
        const CntType count,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto last_input = input[count-1];

    timingKernel(
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count), timer);

    CUDA_FREE(d_temp_storage, mem_stat);
    return output[count-1] + (OutputType)last_input;
}

template<typename DataType, typename SumType, typename CntType>
SumType CUBSum(
        DataType *input,
        CntType count,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    SumType *sum_value = nullptr;
    CUDA_MALLOC(&sum_value, sizeof(SumType), mem_stat);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count), timer);

    SumType res = *sum_value;
    CUDA_FREE(d_temp_storage, mem_stat);
    CUDA_FREE(sum_value, mem_stat);

    return res;
}

template<typename DataType, typename CntType, typename FlagType>
CntType CUBSelect(
        DataType *input,
        DataType *output,
        FlagType *flags,
        const CntType cnt_input,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    CntType *cnt_output = nullptr;
    CUDA_MALLOC(&cnt_output, sizeof(CntType), mem_stat);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, cnt_output, cnt_input), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input, flags, output, cnt_output, cnt_input), timer);

    CUDA_FREE(d_temp_storage, mem_stat);
    auto res = *cnt_output;
    CUDA_FREE(cnt_output, mem_stat);

    return res;
}

template<typename DataType, typename CntType, typename PredicateType>
CntType CUBIf(
        DataType *input,
        DataType *output,
        PredicateType predicate,
        const CntType cnt_input,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    CntType *cnt_output = nullptr;
    CUDA_MALLOC(&cnt_output, sizeof(CntType), mem_stat);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input, output, cnt_output, cnt_input, predicate), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input, output, cnt_output, cnt_input, predicate), timer);

    CUDA_FREE(d_temp_storage, mem_stat);
    auto res = *cnt_output;
    CUDA_FREE(cnt_output, mem_stat);

    return res;
}

template<typename DataType, typename IdxType, typename CntType>
void CUBRadixPartition(
        DataType *input, DataType *output, IdxType *indexes,
        uint32_t bits_begin, uint32_t bits_end,
        const CntType cnt,
        CUDAMemStat *memstat,
        CUDATimeStat *timing)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CntType *asc_indexes_temp;
    CUDA_MALLOC(&asc_indexes_temp, sizeof(CntType)*cnt, memstat);

    thrust::counting_iterator<CntType> iter(0);
    timingKernel(
            thrust::copy(iter, iter + cnt, asc_indexes_temp), timing);

    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input, output, asc_indexes_temp, indexes, cnt, bits_begin, bits_end), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input, output, asc_indexes_temp, indexes, cnt, bits_begin, bits_end), timing);

    CUDA_FREE(asc_indexes_temp, memstat);
    CUDA_FREE(d_temp_storage, memstat);
}

/*Sort pairs (keys,values) according to keys, then values, also provide with the offsets in the original tables*/
template<typename DataType, typename CntType, typename IndexType>
void CUBSortPairs(
        DataType *keysIn, DataType *keysOut,
        DataType *valuesIn, DataType *valuesOut,
        IndexType *idx_ascending, CntType cnt,
        CUDAMemStat *memstat, CUDATimeStat *timing)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CntType *idx_ascending_temp;
    DataType *keys_temp;
    CUDA_MALLOC(&idx_ascending_temp, sizeof(CntType)*cnt, memstat);
    CUDA_MALLOC(&keys_temp, sizeof(DataType)*cnt, memstat);

    thrust::counting_iterator<CntType> iter(0);
    timingKernel(
            thrust::copy(iter, iter + cnt, idx_ascending), timing);

    /*Sort according to values*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, valuesIn, valuesOut, idx_ascending, idx_ascending_temp, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, valuesIn, valuesOut, idx_ascending, idx_ascending_temp, cnt), timing);

    CUDA_FREE(d_temp_storage, memstat);
    d_temp_storage = nullptr;

    /*rearrange the keys*/
    execKernel(gather,GRID_SIZE_DEFAULT,BLOCK_SIZE_DEFAULT,timing,false,keysIn, keys_temp, idx_ascending_temp, cnt);

    /* Sort according to keys, but have to use stable sort*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keysOut, idx_ascending_temp, idx_ascending, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keysOut, idx_ascending_temp, idx_ascending, cnt), timing);

    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(idx_ascending_temp,memstat);
    CUDA_FREE(keys_temp,memstat);

    /*rearrange the values according to the indexes*/
    execKernel(gather,GRID_SIZE_DEFAULT,BLOCK_SIZE_DEFAULT, timing, false, valuesIn, valuesOut, idx_ascending, cnt);
}

template<typename DataType, typename CntType>
CntType CUBUnique(
        DataType *input, DataType *output, CntType count,
        CUDAMemStat *mem_stat, CUDATimeStat *timer)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CntType *num_selected = nullptr;
    CUDA_MALLOC(&num_selected, sizeof(CntType), mem_stat);

    timingKernel(
            cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, input, output, num_selected, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, input, output, num_selected, count), timer);

    auto res = *num_selected;
    CUDA_FREE(num_selected, mem_stat);
    CUDA_FREE(d_temp_storage, mem_stat);

    return res;
}