#include <fstream>
#include <vector>

#include "util/thread_pool.h"
#include "index_query_helper.h"

#include "cuda/primitives.cuh"
#include "cuda/cuda_base.cuh"
#include "cuda/CUDAStat.cuh"

#include "file_loader_gpu.h"
#include "file_loader.h"

#define GetIndexArr GetMallocPReadArrReadOnlyGPU
#define ORDER_MAX_ID (600000010)

using namespace std;

IndexHelper::IndexHelper(string order_path, string line_item_path) {
    auto num_devices = 1;
    cudaGetDeviceCount(&num_devices);
    log_info("Number of GPU devices: %d.", num_devices);

    order_keys_arr.resize(num_devices);
    item_order_keys_arr.resize(num_devices);
    item_prices_arr.resize(num_devices);
    bmp_arr.resize(num_devices);
    dict_arr.resize(num_devices);

    acc_prices_arr.resize(num_devices);
    acc_prices_filtered_arr.resize(num_devices);
    order_offset_arr.resize(num_devices);
    order_offset_filtered_arr.resize(num_devices);

#pragma omp parallel num_threads(num_devices)
    {
        auto gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);
#ifdef UM
        CUDA_MALLOC(&acc_prices_arr[gpu_id], sizeof(double) * MAX_NUM_ORDERS, nullptr);
        CUDA_MALLOC(&acc_prices_filtered_arr[gpu_id], sizeof(double) * MAX_NUM_ORDERS, nullptr);
        CUDA_MALLOC(&order_offset_arr[gpu_id], sizeof(uint32_t) * MAX_NUM_ORDERS, nullptr);
        CUDA_MALLOC(&order_offset_filtered_arr[gpu_id], sizeof(uint32_t) * MAX_NUM_ORDERS, nullptr);
#else
        checkCudaErrors(cudaMalloc((void**)&acc_prices_arr[gpu_id], sizeof(double) * MAX_NUM_ORDERS));
        checkCudaErrors(cudaMalloc((void**)&acc_prices_filtered_arr[gpu_id], sizeof(double) * MAX_NUM_ORDERS));
        checkCudaErrors(cudaMalloc((void**)&order_offset_arr[gpu_id], sizeof(uint32_t) * MAX_NUM_ORDERS));
        checkCudaErrors(cudaMalloc((void**)&order_offset_filtered_arr[gpu_id], sizeof(uint32_t) * MAX_NUM_ORDERS));
#endif
    }

    // Load Order.
    string order_key_path = order_path + ORDER_KEY_BIN_FILE_SUFFIX;
    string order_date_path = order_path + ORDER_DATE_BIN_FILE_SUFFIX;
    string order_meta_path = order_path + ORDER_META_BIN_FILE_SUFFIX;
    {
        ifstream ifs(order_meta_path, std::ifstream::in);
        Archive<ifstream> ar(ifs);

        const char *test_chars = "BUILDING";
        ar >> category_table_ >> min_order_date_ >> max_order_date_
           >> order_second_level_range_ >> order_num_buckets_ >> order_bucket_ptrs_;
        log_info("Probe Test: %d", LinearProbe(category_table_, test_chars, 0, strlen(test_chars)));
    }
    for (auto &category: category_table_) {
        category.PrintStr();
    }
    size_of_orders_ = order_bucket_ptrs_.back();
    num_categories_ = category_table_.size();
    log_info("%d, %d, %d, %d, %d, %zu, %d", num_categories_, min_order_date_, max_order_date_,
             order_second_level_range_, order_num_buckets_, order_bucket_ptrs_.size(), size_of_orders_);
//    cout << order_bucket_ptrs_ << endl;
    int fd;

    ThreadPool pool(num_devices*2);
    vector<future<void>> futures;

    for(auto i = 0; i < num_devices; i++) {
        futures.emplace_back(pool.enqueue([=](){
            int fd;
            auto gpu_id = i;
            cudaSetDevice(gpu_id);
            order_keys_arr[gpu_id] = GetIndexArr<int32_t>(order_key_path.c_str(), fd, size_of_orders_);

            auto &bmp = bmp_arr[gpu_id];
            auto &order_pos_dict = dict_arr[gpu_id];
#ifdef UM
            CUDA_MALLOC(&bmp, sizeof(bool) * ( ORDER_MAX_ID + 1), nullptr);
            CUDA_MALLOC(&order_pos_dict, sizeof(uint32_t) * (ORDER_MAX_ID + 1), nullptr);
#else
            checkCudaErrors(cudaMalloc((void**)&bmp, sizeof(bool) * ( ORDER_MAX_ID + 1)));
            checkCudaErrors(cudaMalloc((void**)&order_pos_dict, sizeof(uint32_t) * (ORDER_MAX_ID + 1)));
#endif
        }));
    }

    order_dates_ = GetMMAPArrReadOnly<uint32_t>(order_date_path.c_str(), fd, size_of_orders_);
    log_info("Finish Order Index Loading...Not Populate Yet");

    // Load LineItem.
    string item_order_id_path = line_item_path + LINE_ITEM_ORDER_KEY_FILE_SUFFIX;
    string item_price_path = line_item_path + LINE_ITEM_PRICE_FILE_SUFFIX;
    string item_meta_path = line_item_path + LINE_ITEM_META_BIN_FILE_SUFFIX;
    {
        ifstream ifs(item_meta_path, std::ifstream::in);
        Archive<ifstream> ar(ifs);
        ar >> min_ship_date_ >> max_ship_date_ >> item_num_buckets_ >> item_bucket_ptrs_;
    }
    size_of_items_ = item_bucket_ptrs_.back();
    log_info("%d, %d, %d, %zu, %d", min_ship_date_, max_ship_date_, item_num_buckets_, item_bucket_ptrs_.size(),
             size_of_items_);

    for(auto i = 0; i < num_devices; i++) {
        futures.emplace_back(pool.enqueue([=](){
            int fd = 0;
            auto gpu_id = i;
            cudaSetDevice(gpu_id);
            item_order_keys_arr[gpu_id] = GetIndexArr<int32_t>(item_order_id_path.c_str(), fd, size_of_items_);
            item_prices_arr[gpu_id] = GetIndexArr<double>(item_price_path.c_str(), fd, size_of_items_);
        }));
    }
    for(auto &future: futures) {
        future.get();
    }

    log_info("Finish LineItem Loading...Not Populate Yet");
}

__global__
void buildBooleanArray(
        uint32_t start_pos, uint32_t end_pos,
        int32_t *order_keys_, bool *bmp, uint32_t *order_pos_dict) {
    auto gtid = threadIdx.x + blockDim.x * blockIdx.x + start_pos;
    auto gtnum = blockDim.x * gridDim.x;

    while (gtid < end_pos) {
        auto order_key = order_keys_[gtid];
        bmp[order_key] = true;
        order_pos_dict[order_key] = gtid - start_pos;
        gtid += gtnum;
    }
}

__global__
void filterJoin(
        uint32_t start_pos, uint32_t end_pos,
        int32_t *item_order_keys_, double *acc_prices, double *item_prices_, int32_t max_order_id,
        bool *bmp, uint32_t *order_pos_dict) {
    auto gtid = threadIdx.x + blockDim.x * blockIdx.x + start_pos;
    auto gtnum = blockDim.x * gridDim.x;

    while (gtid < end_pos) {
        auto order_key = item_order_keys_[gtid];
        if ((order_key <= max_order_id) && (bmp[order_key])) {
            atomicAdd(&acc_prices[order_pos_dict[order_key]], item_prices_[gtid]);
        }
        gtid += gtnum;
    }
}

void IndexHelper::evaluateWithGPU(
        vector<int32_t *> order_keys_arr, uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
        vector<int32_t *> item_order_keys_arr, uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
        vector<bool*> bmp_arr, vector<uint32_t *> dict_arr,
        vector<double *> item_prices_arr, uint32_t order_array_view_size, int lim, int32_t &size_of_results, Result *t) {
    CUDAMemStat memstat_detail;
    CUDATimeStat timing_detail;
    auto memstat = &memstat_detail;
    auto timing = &timing_detail;

    log_trace("Evaluate with GPUs");

    auto num_devices = 1;
    cudaGetDeviceCount(&num_devices);
    log_info("Number of GPU devices: %d.", num_devices);

    auto lineitem_tuples_per_gpu = (lineitem_bucket_ptr_end - lineitem_bucket_ptr_beg + num_devices - 1) / num_devices;

    Timer timer;

    /*compute max_order_id with a single GPU*/
    cudaSetDevice(0);
    int32_t max_order_id = CUBMax(&order_keys_arr[0][order_bucket_ptr_beg], (order_bucket_ptr_end - order_bucket_ptr_beg),
                                  memstat, timing);
    log_info("BMP Size: %d", max_order_id + 1);
    log_info("After get max_order_id: %.2f s.", timer.elapsed());

    cudaDeviceSynchronize();

#pragma omp parallel num_threads(num_devices)
    {
        auto gpu_id = omp_get_thread_num();
        log_info("TID: %d, BMP Size: %d", gpu_id, max_order_id + 1);

        cudaSetDevice(gpu_id);

        auto lineitem_bucket_ptr_beg_gpu = lineitem_bucket_ptr_beg + gpu_id * lineitem_tuples_per_gpu;
        auto lineitem_bucket_ptr_end_gpu = lineitem_bucket_ptr_beg + (gpu_id+1) * lineitem_tuples_per_gpu;
        if (lineitem_bucket_ptr_end_gpu > lineitem_bucket_ptr_end)
            lineitem_bucket_ptr_end_gpu = lineitem_bucket_ptr_end;
        log_info("GPU ID: %d, lineitem range: [%d, %d)", gpu_id, lineitem_bucket_ptr_beg_gpu, lineitem_bucket_ptr_end_gpu);

        checkCudaErrors(cudaMemset(acc_prices_arr[gpu_id], 0, sizeof(double) * order_array_view_size));
        log_info("After malloc acc_prices_arr: %.2f s.", timer.elapsed());

        /*construct the mapping*/
        auto bmp = bmp_arr[gpu_id];
        auto order_pos_dict = dict_arr[gpu_id];
        checkCudaErrors(cudaMemset(bmp, 0, sizeof(bool) * (max_order_id + 1)));

        log_info("TID: %d, Before Construction Data Structures: %.6lfs", gpu_id, timer.elapsed());

        /*build the boolean filter*/
        execKernel(buildBooleanArray, 1024, 256, timing, false,
                   order_bucket_ptr_beg, order_bucket_ptr_end,
                   order_keys_arr[gpu_id], bmp, order_pos_dict);

        log_info("TID: %d, Before Aggregation: %.6lfs", gpu_id,  timer.elapsed());

        execKernel(filterJoin, 1024, 256, timing, false,
                   lineitem_bucket_ptr_beg_gpu, lineitem_bucket_ptr_end_gpu,
                   item_order_keys_arr[gpu_id], acc_prices_arr[gpu_id], item_prices_arr[gpu_id], max_order_id, bmp, order_pos_dict);

        if (gpu_id != 0) //prefetch to gpu0
        {
#ifdef UM
            checkCudaErrors(cudaMemPrefetchAsync(acc_prices_arr[gpu_id], sizeof(double)*order_array_view_size, 0));
#endif
        }
        log_info("TID: %d, Before Select: %.6lfs", gpu_id, timer.elapsed());
    }

    /*add up the acc_prices*/
    auto iter_begin = thrust::make_counting_iterator(0u);
    auto iter_end = thrust::make_counting_iterator(order_array_view_size);

    cudaSetDevice(0);
    for(auto i = 1; i < num_devices; i++) {
        double *acc_prices_0 = acc_prices_arr[0];
        double *acc_prices_i = acc_prices_arr[i];
        timingKernel(
                thrust::transform(thrust::device, iter_begin, iter_end, acc_prices_arr[0], [=]
                __device__(uint32_t idx) {
                return acc_prices_0[idx] + acc_prices_i[idx];
        }), timing);
    }

    /*processing the summarized acc_prices*/
    bool *flag_is_zero = nullptr;
#ifdef UM
    CUDA_MALLOC(&flag_is_zero, sizeof(bool) * order_array_view_size, memstat);
#else
    checkCudaErrors(cudaMalloc(&flag_is_zero, sizeof(bool) * order_array_view_size));
#endif

    /*the acc_price double buffer*/
    auto *acc_price_temp = acc_prices_filtered_arr[0];
    auto *acc_prices = acc_prices_arr[0];
    log_info("Before processing the summarized acc_prices: %.2f s.", timer.elapsed());

    /*the order_offset double buffer*/
    auto *order_offset = order_offset_arr[0];
    auto *order_offset_temp = order_offset_filtered_arr[0];

    thrust::counting_iterator<uint32_t> iter(order_bucket_ptr_beg);
    timingKernel(
            thrust::copy(iter, iter + order_array_view_size, order_offset), timing);

    /*construct the boolean filter*/
    timingKernel(
            thrust::transform(thrust::device, iter_begin, iter_end, flag_is_zero, [=]
            __device__(uint32_t idx) {
            return acc_prices[idx] > 0.0;
    }), timing);

    /*filter the acc_price*/
    size_of_results = CUBSelect(acc_prices, acc_price_temp, flag_is_zero, order_array_view_size, memstat, timing);
    CUBSelect(order_offset, order_offset_temp, flag_is_zero, order_array_view_size, memstat, timing);

    log_info("Non Zeros: %zu", size_of_results);

    /*CUB sort pairs*/
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, acc_price_temp, acc_prices, order_offset_temp, order_offset, size_of_results);
#ifdef UM
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
#else
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
#endif
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, acc_price_temp, acc_prices, order_offset_temp, order_offset, size_of_results);
    checkCudaErrors(cudaDeviceSynchronize());

    for (auto i = 0; i < lim; i++) {
        t[i].price = acc_prices[i];
        t[i].order_offset = order_offset[i];
    }

    log_info("Maximal device memory demanded: %ld bytes.", memstat->get_max_use());
    log_info("Unfreed device memory size: %ld bytes.", memstat->get_cur_use());
}