#include <fstream>
#include <vector>

#include "util/thread_pool.h"
#include "index_query_helper.h"

#include "cuda/primitives.cuh"
#include "cuda/cuda_base.cuh"
#include "cuda/CUDAStat.cuh"

#include "file_loader.h"
#include "util/primitives/parasort_cmp.h"

#define GetIndexArr GetMallocPReadArrReadOnly

using namespace std;

IndexHelper::IndexHelper(string order_path, string line_item_path) {
    auto num_devices = 1;
    cudaGetDeviceCount(&num_devices);
    log_info("Number of GPU devices: %d.", num_devices);

    order_keys_arr_.resize(num_devices);
    item_order_keys_arr_.resize(num_devices);
    bmp_arr_.resize(num_devices);
    matches_.resize(num_devices);
    num_matches_.resize(num_devices);

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

    order_keys_ = GetIndexArr<uint32_t>(order_key_path.c_str(), fd, size_of_orders_);

    for(auto i = 0; i < num_devices; i++) {
        futures.emplace_back(pool.enqueue([=](){
            auto gpu_id = i;
            cudaSetDevice(gpu_id);

            /*load order_keys to GPU device mem*/
            auto &d_arr = order_keys_arr_[gpu_id];
            checkCudaErrors(cudaMalloc((void**)&d_arr, sizeof(uint32_t)*size_of_orders_));
            checkCudaErrors(cudaMemcpy(d_arr, order_keys_, sizeof(uint32_t)*size_of_orders_, cudaMemcpyHostToDevice));

            auto &bmp = bmp_arr_[gpu_id];
            size_t bitmap_cnt = ((ORDER_MAX_ID+1)+sizeof(uint32_t)*8-1)/ sizeof(uint32_t)/8;
            checkCudaErrors(cudaMalloc((void**)&bmp, sizeof(uint32_t)*bitmap_cnt));
            log_trace("Allocated bitmap size: %lu bytes (no UM).", sizeof(uint32_t)*bitmap_cnt);
        }));
    }

    /*Load the remaining CPU arrays*/
    order_dates_ = GetMMAPArrReadOnly<uint32_t>(order_date_path.c_str(), fd, size_of_orders_);
    log_info("Finish Order Index Loading...Not Populate Yet");

    // Load LineItem.
    string item_order_id_path = line_item_path + LINE_ITEM_ORDER_KEY_FILE_SUFFIX;
    string item_price_path = line_item_path + LINE_ITEM_PRICE_FILE_SUFFIX;
    string item_meta_path = line_item_path + LINE_ITEM_META_BIN_FILE_SUFFIX;
    {
        ifstream ifs(item_meta_path, std::ifstream::in);
        Archive<ifstream> ar(ifs);
        ar >> max_order_id_ >> min_ship_date_ >> max_ship_date_ >> item_num_buckets_ >> item_bucket_ptrs_;
    }
    size_of_items_ = item_bucket_ptrs_.back();
    log_info("%d, %d, %d, %zu, %u", min_ship_date_, max_ship_date_, item_num_buckets_, item_bucket_ptrs_.size(), size_of_items_);

    item_order_keys_ = GetIndexArr<uint32_t>(item_order_id_path.c_str(), fd, size_of_items_);
    for(auto i = 0; i < num_devices; i++) {
        futures.emplace_back(pool.enqueue([=](){
            auto gpu_id = i;
            cudaSetDevice(gpu_id);

            /*load lineitem_order_keys to GPU device mem*/
            auto &d_arr = item_order_keys_arr_[gpu_id];
            checkCudaErrors(cudaMalloc((void**)&d_arr, sizeof(uint32_t)*size_of_items_));
            checkCudaErrors(cudaMemcpy(d_arr, item_order_keys_, sizeof(uint32_t)*size_of_items_, cudaMemcpyHostToDevice));
        }));
    }
    for(auto &future: futures) {
        future.get();
    }

    /*Load the remaining CPU arrays*/
    item_prices_ = GetMMAPArrReadOnly<double>(item_price_path.c_str(), fd, size_of_items_);
    log_info("Finish LineItem Loading...Not Populate Yet");
}

__global__
void buildBitmap(
        uint32_t start_pos, uint32_t end_pos,
        uint32_t *order_keys_, uint32_t *bmp) {
    auto gtid = threadIdx.x + blockDim.x * blockIdx.x + start_pos;
    auto gtnum = blockDim.x * gridDim.x;

    while (gtid < end_pos) {
        auto order_key = order_keys_[gtid];
        auto byte_pos = order_key >> 5;
        auto offset_in_byte = order_key & 31;
        atomicOr(&bmp[byte_pos], (uint32_t)(1<<offset_in_byte));
        gtid += gtnum;
    }
}

/*only retrieve the offsets of the matching tuples in the lineitem table*/
__global__
void bitmapJoinCnt(
        uint32_t start_pos, uint32_t end_pos,
        uint32_t *item_order_keys_, uint32_t max_order_id,
        uint32_t *bmp, uint32_t *cnts) {
    auto gtid = threadIdx.x + blockDim.x * blockIdx.x + start_pos;
    auto gtnum = blockDim.x * gridDim.x;
    __shared__ uint32_t lcnt;

    if (threadIdx.x == 0) lcnt = 0;
    __syncthreads();

    while (gtid < end_pos) {
        auto order_key = item_order_keys_[gtid];
        auto byte_pos = order_key >> 5;
        auto offset_in_byte = order_key & 31;

        if ((order_key <= max_order_id) && ((bmp[byte_pos] >> offset_in_byte) & 0b1)) {
            atomicAdd(&lcnt, 1);
        }
        gtid += gtnum;
    }
    __syncthreads();

    if (threadIdx.x == 0)
        cnts[blockIdx.x] = lcnt;
}

__global__
void bitmapJoinWrt(
        uint32_t start_pos, uint32_t end_pos,
        uint32_t *item_order_keys_, uint32_t max_order_id,
        uint32_t *bmp, uint32_t *cnts, uint32_t *matches) {
    auto gtid = threadIdx.x + blockDim.x * blockIdx.x + start_pos;
    auto gtnum = blockDim.x * gridDim.x;
    __shared__ uint32_t lpos;

    if (threadIdx.x == 0) lpos = cnts[blockIdx.x];
    __syncthreads();

    while (gtid < end_pos) {
        auto order_key = item_order_keys_[gtid];
        auto byte_pos = order_key >> 5;
        auto offset_in_byte = order_key & 31;

        if ((order_key <= max_order_id) && ((bmp[byte_pos] >> offset_in_byte) & 0b1)) {
            auto cur = atomicAdd(&lpos, 1);
            matches[cur] = gtid;
        }
        gtid += gtnum;
    }
}

inline uint64_t __double_as_longlong_CPU(double input) {
    union {
        uint64_t res;
        double src;
    };
    src = input;
    return res;
}

inline double __longlong_as_double_CPU(uint64_t input) {
    union {
        double res;
        uint64_t src;
    };
    src = input;
    return res;
}

inline double __sync_fetch_and_add_double(double *address, double val) {
    uint64_t *address_as_ull = (uint64_t *) address;
    uint64_t old = *address_as_ull;
    uint64_t assumed;

    do {
        assumed = old;
        old = __sync_val_compare_and_swap(address_as_ull, assumed,
                                          __double_as_longlong_CPU(val + __longlong_as_double_CPU(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double_CPU(assumed);
}

void IndexHelper::evaluateWithGPU(
        uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
        uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
        uint32_t order_array_view_size, int lim,
        uint32_t &size_of_results, Result *results) {
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
    uint32_t max_order_id = CUBMax(&order_keys_arr_[0][order_bucket_ptr_beg], (order_bucket_ptr_end - order_bucket_ptr_beg), memstat, timing);
    log_info("max_order_id: %lu, max_order_id_: %lu.", max_order_id, max_order_id_);
    log_info("BMP Size: %u bytes", (max_order_id + 1 + 7)/8 );
    log_info("After get max_order_id: %.2f s.", timer.elapsed());

    checkCudaErrors(cudaDeviceSynchronize());

#pragma omp parallel num_threads(num_devices)
    {
        auto gpu_id = omp_get_thread_num();
        log_info("TID: %d, BMP Size: %u bytes", gpu_id, (max_order_id + 1 + 7)/8 );

        cudaSetDevice(gpu_id);
        auto lineitem_bucket_ptr_beg_gpu = lineitem_bucket_ptr_beg + gpu_id * lineitem_tuples_per_gpu;
        auto lineitem_bucket_ptr_end_gpu = lineitem_bucket_ptr_beg + (gpu_id+1) * lineitem_tuples_per_gpu;
        if (lineitem_bucket_ptr_end_gpu > lineitem_bucket_ptr_end)
            lineitem_bucket_ptr_end_gpu = lineitem_bucket_ptr_end;
        log_info("GPU ID: %d, lineitem range: [%u, %u)", gpu_id, lineitem_bucket_ptr_beg_gpu, lineitem_bucket_ptr_end_gpu);

        /*construct the mapping*/
        auto bmp = bmp_arr_[gpu_id];
        checkCudaErrors(cudaMemset(bmp, 0, sizeof(bool) * (max_order_id + 1 + 7)/8));

        log_info("TID: %d, Before Construction Data Structures: %.6lfs", gpu_id, timer.elapsed());

        /*build the bitmap*/
        execKernel(buildBitmap, 1024, 256, timing, false,
                   order_bucket_ptr_beg, order_bucket_ptr_end,
                   order_keys_arr_[gpu_id], bmp);

        log_info("TID: %d, Before join count: %.6lfs", gpu_id,  timer.elapsed());

        int grid_size = 1024, block_size = 256;
        uint32_t *cnts = nullptr;
        CUDA_MALLOC(&cnts, sizeof(uint32_t)*grid_size, nullptr);

        execKernel(bitmapJoinCnt, grid_size, block_size, timing, false,
           lineitem_bucket_ptr_beg_gpu, lineitem_bucket_ptr_end_gpu,
           item_order_keys_arr_[gpu_id], max_order_id, bmp, cnts);

        num_matches_[gpu_id] = CUBScanExclusive(cnts, cnts, grid_size, memstat, timing);

        log_info("TID: %d, #matching items: %d.", gpu_id, num_matches_[gpu_id]);
        log_info("TID: %d, Before join write: %.6lfs", gpu_id, timer.elapsed());

        if (num_matches_[gpu_id] > 0) {
            CUDA_MALLOC(&matches_[gpu_id], sizeof(uint32_t)*num_matches_[gpu_id], nullptr); //use unified memory
            execKernel(bitmapJoinWrt, grid_size, block_size, timing, false,
                       lineitem_bucket_ptr_beg_gpu, lineitem_bucket_ptr_end_gpu,
                       item_order_keys_arr_[gpu_id], max_order_id, bmp, cnts, matches_[gpu_id]);

            /*prefetched to CPU in advance*/
            checkCudaErrors(cudaMemPrefetchAsync(matches_[gpu_id], sizeof(uint32_t)*num_matches_[gpu_id], cudaCpuDeviceId));
        }
        log_info("TID: %d, Before Select: %.6lfs", gpu_id, timer.elapsed());
    }

    log_info("Parallel processing time: %.2f s.", timer.elapsed());

    /*CPU processing*/
    auto acc_prices = (double *) malloc(sizeof(double) * order_array_view_size);
    auto order_pos_dict = (uint32_t *) malloc(sizeof(uint32_t) * (max_order_id_ + 1));
    auto relative_off = (uint32_t *) malloc(sizeof(uint32_t) * order_array_view_size);
    vector<uint32_t> histogram;

#pragma omp parallel
    {
        MemSetOMP(acc_prices, 0, order_array_view_size);
#pragma omp single
        log_info("Before set dict: %.6lfs", timer.elapsed());

#pragma omp for
        for(auto i = order_bucket_ptr_beg; i < order_bucket_ptr_end; i++) {
            auto order_key = order_keys_[i];
            order_pos_dict[order_key] = i - order_bucket_ptr_beg;
        }
#pragma omp single
        log_info("Before adding up the price: %.6lfs", timer.elapsed());

        /*process the results computed by each GPU*/
        for(auto d = 0; d < num_devices; d++) {
#pragma omp for
            for(auto i = 0; i < num_matches_[d]; i++) {
                auto offset = matches_[d][i];
                auto order_key = item_order_keys_[offset];
                auto price = item_prices_[offset];
                __sync_fetch_and_add_double(&acc_prices[order_pos_dict[order_key]], price);
            }
        }
#pragma omp single
        log_info("Before filtering away zero acc_prices: %.6lfs", timer.elapsed());

        FlagPrefixSumOMP(histogram, relative_off, order_array_view_size, [acc_prices](uint32_t it) {
            return acc_prices[it] == 0;
        });
#pragma omp single
        log_info("Before producing results: %.6lfs", timer.elapsed());
#pragma omp for reduction(+:size_of_results)
        for (uint32_t i = 0u; i < order_array_view_size; i++) {
            if (acc_prices[i] != 0) {
                size_of_results++;
                auto off = i - relative_off[i];
                results[off] = {.order_offset=i + order_bucket_ptr_beg, .price= acc_prices[i]};
            }
        }
    }

    log_info("Non Zeros: %zu", size_of_results);
    parasort(size_of_results, results, [](Result l, Result r) {
        return l.price > r.price;
    }, omp_get_max_threads());
}