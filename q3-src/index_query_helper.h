#pragma once

#include <cstdint>
#include <vector>

#include "config.h"
#include "file_parser.h"

using namespace std;

struct Result {
    uint32_t order_offset;
    double price;
};

class IndexHelper {
    // Customer.
    vector<String> category_table_;
    uint32_t num_categories_ = 0;

    // Order.
    uint32_t max_order_date_ = 0;
    uint32_t min_order_date_ = UINT32_MAX;
    uint32_t order_second_level_range_ = 0;
    uint32_t order_num_buckets_ = 0;
    uint32_t size_of_orders_ = 0;

    // Order CSR.
    vector<uint32_t> order_bucket_ptrs_;
    int32_t *order_keys_ = nullptr;
    uint32_t *order_dates_ = nullptr;

    // LineItem.
    uint32_t max_ship_date_ = 0;
    uint32_t min_ship_date_ = UINT32_MAX;
    uint32_t item_num_buckets_ = 0;
    uint32_t size_of_items_ = 0;

    // Item CSR.
    vector<uint32_t> item_bucket_ptrs_;
    int32_t *item_order_keys_ = nullptr;
    double *item_prices_ = nullptr;

    //vectors for multi-GPU
    vector<int32_t*> order_keys_arr;
    vector<int32_t*> item_order_keys_arr;
    vector<double*> item_prices_arr;

public:
    IndexHelper(string order_path, string line_item_path);

    void Query(string category, string order_date, string ship_date, int limit);

    vector<bool*> bmp_arr;
    vector<uint32_t *> dict_arr;
    vector<double*> acc_prices_arr;
    vector<double*> acc_prices_filtered_arr;

    vector<uint32_t*> order_offset_arr;
    vector<uint32_t*> order_offset_filtered_arr;

    void evaluateWithGPU(
            vector<int32_t *> order_keys_arr, uint32_t order_bucket_ptr_beg, uint32_t order_bucket_ptr_end,
            vector<int32_t *> item_order_keys_arr, uint32_t lineitem_bucket_ptr_beg, uint32_t lineitem_bucket_ptr_end,
            vector<bool*> bmp_arr, vector<uint32_t *> dict_arr,
            vector<double *> item_prices_arr, uint32_t order_array_view_size, int lim, int32_t &size_of_results, Result *t);
};

