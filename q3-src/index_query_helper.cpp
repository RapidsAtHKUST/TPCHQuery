//
// Created by yche on 10/20/19.
//

#include "index_query_helper.h"

#include <string>
#include <fstream>

#include "file_loader.h"
#include "util/archive.h"
#include "util/pretty_print.h"
#include "util/primitives/parasort_cmp.h"

using namespace std;

#ifndef USE_GPU
#define GetIndexArr GetMallocPReadArrReadOnly
//#define GetIndexArr GetMMAPArrReadOnly
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
#endif

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

void evaluateWithCPU(
        int32_t *order_keys_, uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
        int32_t *item_order_keys_, uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
        double *item_prices_, uint32_t order_array_view_size, int lim,
        int32_t &size_of_results, Result *results) {
    log_trace("Evaluate with CPU");

    Timer timer;
    auto relative_off = (uint32_t *) malloc(sizeof(uint32_t) * order_array_view_size);
    uint32_t *order_pos_dict;
    bool *bmp;
    int32_t max_order_id = 0;
    vector<uint32_t> histogram;

    auto acc_prices = (double *) malloc(sizeof(double) * order_array_view_size);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        MemSetOMP(acc_prices, 0, order_array_view_size, tid, num_threads);
//#pragma omp for
//        for (size_t i = 0; i < order_array_view_size; i++) {
//            assert(acc_prices[i] == 0);
//        }
#pragma omp for reduction(max:max_order_id)
        for (size_t i = order_bucket_ptr_beg; i < order_bucket_ptr_end; i++) {
            max_order_id = max(max_order_id, order_keys_[i]);
        }
#pragma omp single
        {
            log_info("BMP Size: %d", max_order_id + 1);
            bmp = (bool *) malloc(sizeof(bool) * (max_order_id + 1));
            order_pos_dict = (uint32_t *) malloc(sizeof(uint32_t) * (max_order_id + 1));
        }
        MemSetOMP(bmp, 0, (max_order_id + 1), tid, num_threads);
#pragma omp single
        log_info("Before Construction Data Structures: %.6lfs", timer.elapsed());
#pragma omp for
        for (auto i = order_bucket_ptr_beg; i < order_bucket_ptr_end; i++) {
            auto order_key = order_keys_[i];
            bmp[order_key] = true;
            order_pos_dict[order_key] = i - order_bucket_ptr_beg;
        }
#pragma omp single
        log_info("Before Aggregation: %.6lfs", timer.elapsed());
#pragma omp for
        for (size_t i = lineitem_bucket_ptr_beg; i < lineitem_bucket_ptr_end; i++) {
            auto order_key = item_order_keys_[i];
            if ((order_key <= max_order_id) && (bmp[order_key])) {
                __sync_fetch_and_add_double(&acc_prices[order_pos_dict[order_key]], item_prices_[i]);
            }
        }
#pragma omp single
        log_info("Before Select: %.6lfs", timer.elapsed());
        FlagPrefixSumOMP(histogram, relative_off, order_array_view_size, [acc_prices](uint32_t it) {
            return acc_prices[it] == 0;
        }, num_threads);
#pragma omp for reduction(+:size_of_results)
        for (uint32_t i = 0u; i < order_array_view_size; i++) {
            if (acc_prices[i] != 0) {
                size_of_results++;
                auto off = i - relative_off[i];
                results[off] = {.order_offset=i + order_bucket_ptr_beg, .price= acc_prices[i]};
            }
        }
    }
    free(order_pos_dict);
    free(bmp);
    free(acc_prices);
    free(relative_off);
    log_info("Non Zeros: %zu", size_of_results);
    // Select & Sort & Return Top (min(K, #results)).
#ifdef BASELINE_SORT
    sort(results, results + size_of_results, [](Result l, Result r) {
        return l.price > r.price;
    });
#else
    parasort(size_of_results, results, [](Result l, Result r) {
        return l.price > r.price;
    }, omp_get_max_threads());
#endif
}

void evaluateWithGPU(
        vector<int32_t *> order_keys_arr, uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
        vector<int32_t *> item_order_keys_arr, uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
        vector<bool*> bmp_arr, vector<uint32_t *> dict_arr,
        vector<double *> item_prices_arr, uint32_t order_array_view_size, int lim, int32_t &size_of_results, Result *t);

void IndexHelper::Query(string category, string order_date, string ship_date, int limit) {
    uint32_t o_date = ConvertDateToBucketID(order_date.c_str());
    uint32_t s_date = ConvertDateToBucketID(ship_date.c_str());
    int category_id = LinearProbe(category_table_, category.c_str(), 0, category.size());
    log_info("%d, %.*s, [%d, %d), [%d, %d)", category_id, category_table_[category_id].size,
             category_table_[category_id].chars,
             min_order_date_, o_date, s_date + 1, max_ship_date_ + 1);
    // Exclusive "< o_date", "> s_date".
    if (o_date <= min_order_date_ || s_date >= max_ship_date_) {
        return;
    }
    uint32_t o_bucket_beg = category_id * order_second_level_range_ + 0;
    uint32_t o_bucket_end = category_id * order_second_level_range_ +
                            (min(o_date, max_order_date_ + 1) - min_order_date_);

    uint32_t item_bucket_beg = max(s_date + 1, min_ship_date_) - min_ship_date_;
    uint32_t item_bucket_end = item_num_buckets_;
    log_info("Order Bucket Range: [%d, %d); Item Bucket Range: [%d, %d)",
             o_bucket_beg, o_bucket_end, item_bucket_beg, item_bucket_end);
    auto order_array_view_size = order_bucket_ptrs_[o_bucket_end] - order_bucket_ptrs_[o_bucket_beg];
    auto item_array_view_size = item_bucket_ptrs_[item_bucket_end] - item_bucket_ptrs_[item_bucket_beg];
    log_info("[%d, %d): %d, [%d, %d): %d", order_bucket_ptrs_[o_bucket_beg], order_bucket_ptrs_[o_bucket_end],
             order_array_view_size, item_bucket_ptrs_[item_bucket_beg], item_bucket_ptrs_[item_bucket_end],
             item_array_view_size);

    // Join & Aggregate.
    auto results = (Result *) malloc(sizeof(Result) * order_array_view_size);
    int32_t size_of_results = 0;
    log_info("%d", order_keys_arr[0][order_bucket_ptrs_[o_bucket_beg]]);

    Timer timer;
    auto order_bucket_ptr_beg = order_bucket_ptrs_[o_bucket_beg];
    auto order_bucket_ptr_end = order_bucket_ptrs_[o_bucket_end];

#ifdef USE_GPU
    evaluateWithGPU(
            order_keys_arr, order_bucket_ptr_beg, order_bucket_ptr_end,
            item_order_keys_arr, item_bucket_ptrs_[item_bucket_beg], item_bucket_ptrs_[item_bucket_end],
            bmp_arr,  dict_arr,
            item_prices_arr, order_array_view_size, limit, size_of_results, results);
#else
    evaluateWithCPU(
            order_keys_, order_bucket_ptr_beg, order_bucket_ptr_end,
            item_order_keys_, item_bucket_ptrs_[item_bucket_beg], item_bucket_ptrs_[item_bucket_end],
            item_prices_, order_array_view_size, limit, size_of_results, results);
#endif

    printf("l_orderkey|o_orderdate|revenue\n");
    for (auto i = 0; i < min<int32_t>(size_of_results, limit); i++) {
        char date[DATE_LEN + 1];
        date[DATE_LEN] = '\0';
        ConvertBucketIDToDate(date, order_dates_[results[i].order_offset]);
        printf("%d|%s|%.2lf\n",
               order_keys_arr[0][results[i].order_offset], date, results[i].price);
    }
    log_info("Query Time: %.6lfs", timer.elapsed());
}
