#include <fstream>

#include "index_query_helper.h"

#include "cuda/primitives.cuh"
#include "cuda/cuda_base.cuh"
#include "cuda/CUDAStat.cuh"

#include "file_loader_gpu.h"
#include "file_loader.h"

#define GetIndexArr GetMallocPReadArrReadOnlyGPU

using namespace std;

IndexHelper::IndexHelper(string order_path, string line_item_path) {
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
    order_keys_ = GetIndexArr<int32_t>(order_key_path.c_str(), fd, size_of_orders_);
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
    item_order_keys_ = GetIndexArr<int32_t>(item_order_id_path.c_str(), fd, size_of_items_);
    item_prices_ = GetIndexArr<double>(item_price_path.c_str(), fd, size_of_items_);
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

void evaluateWithGPU(
        int32_t *order_keys_, uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
        int32_t *item_order_keys_, uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
        double *item_prices_, uint32_t order_array_view_size, int lim, int32_t &size_of_results, Result *t) {
    CUDAMemStat memstat_detail;
    CUDATimeStat timing_detail;
    auto memstat = &memstat_detail;
    auto timing = &timing_detail;

    log_trace("Evaluate with GPU");

    Timer timer;
    int32_t max_order_id = CUBMax(&order_keys_[order_bucket_ptr_beg], (order_bucket_ptr_end - order_bucket_ptr_beg),
                                  memstat, timing);
    log_info("BMP Size: %d", max_order_id + 1);

    double *acc_prices = nullptr;
    CUDA_MALLOC(&acc_prices, sizeof(double) * order_array_view_size, memstat);
    checkCudaErrors(cudaMemset(acc_prices, 0, sizeof(double) * order_array_view_size));

    /*construct the mapping*/
    bool *bmp = nullptr;
    uint32_t *order_pos_dict = nullptr;
    CUDA_MALLOC(&bmp, sizeof(bool) * (max_order_id + 1), memstat);
    CUDA_MALLOC(&order_pos_dict, sizeof(uint32_t) * (max_order_id + 1), memstat);
    checkCudaErrors(cudaMemset(bmp, 0, sizeof(bool) * (max_order_id + 1)));
    log_info("Before Construction Data Structures: %.6lfs", timer.elapsed());

    /*build the boolean filter*/
    execKernel(buildBooleanArray, 1024, 256, timing, false,
               order_bucket_ptr_beg, order_bucket_ptr_end,
               order_keys_, bmp, order_pos_dict);
    log_info("Before Aggregation: %.6lfs", timer.elapsed());

    execKernel(filterJoin, 1024, 256, timing, false,
               lineitem_bucket_ptr_beg, lineitem_bucket_ptr_end,
               item_order_keys_, acc_prices, item_prices_, max_order_id, bmp, order_pos_dict);
    log_info("Before Select: %.6lfs", timer.elapsed());

    bool *flag_is_zero = nullptr;
    CUDA_MALLOC(&flag_is_zero, sizeof(bool) * order_array_view_size, memstat);

    /*the results*/
    double *acc_price_filtered = nullptr;
    CUDA_MALLOC(&acc_price_filtered, sizeof(double) * order_array_view_size, memstat);

    uint32_t *order_offset = nullptr, *order_offset_filtered = nullptr;
    CUDA_MALLOC(&order_offset, sizeof(uint32_t) * order_array_view_size, memstat);
    CUDA_MALLOC(&order_offset_filtered, sizeof(uint32_t) * order_array_view_size, memstat);

    auto iter_begin = thrust::make_counting_iterator(0u);
    auto iter_end = thrust::make_counting_iterator(order_array_view_size);

    thrust::counting_iterator<uint32_t> iter(order_bucket_ptr_beg);
    timingKernel(
            thrust::copy(iter, iter + order_array_view_size, order_offset), timing);

    /*construct the boolean filter*/
    timingKernel(
            thrust::transform(thrust::device, iter_begin, iter_end, flag_is_zero, [=]
            __device__(uint32_t
            idx) {
            return acc_prices[idx] > 0.0;
    }), timing);

    /*filter the acc_price*/
    size_of_results = CUBSelect(acc_prices, acc_price_filtered, flag_is_zero, order_array_view_size, memstat, timing);
    CUBSelect(order_offset, order_offset_filtered, flag_is_zero, order_array_view_size, memstat, timing);

    CUDA_FREE(flag_is_zero, memstat);
    CUDA_FREE(order_offset, memstat);
    CUDA_FREE(acc_prices, memstat);

    log_info("Non Zeros: %zu", size_of_results);

    double *acc_price_sorted = nullptr;
    uint32_t *order_offset_sorted = nullptr;
    CUDA_MALLOC(&acc_price_sorted, sizeof(double) * size_of_results, memstat);
    CUDA_MALLOC(&order_offset_sorted, sizeof(uint32_t) * size_of_results, memstat);

    /*CUB sort pairs*/
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, acc_price_filtered, acc_price_sorted,
                                              order_offset_filtered, order_offset_sorted, size_of_results);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, acc_price_filtered, acc_price_sorted,
                                              order_offset_filtered, order_offset_sorted, size_of_results);
    cudaDeviceSynchronize();

    for (auto i = 0; i < lim; i++) {
        t[i].price = acc_price_sorted[i];
        t[i].order_offset = order_offset_sorted[i];
    }

    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(bmp, memstat);
    CUDA_FREE(order_pos_dict, memstat);
    CUDA_FREE(acc_price_filtered, memstat);
    CUDA_FREE(order_offset_filtered, memstat);
    CUDA_FREE(acc_price_sorted, memstat);
    CUDA_FREE(order_offset_sorted, memstat);

    log_info("Maximal device memory demanded: %ld bytes.", memstat->get_max_use());
    log_info("Unfreed device memory size: %ld bytes.", memstat->get_cur_use());
}