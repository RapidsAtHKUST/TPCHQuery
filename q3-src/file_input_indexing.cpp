//
// Created by yche on 10/20/19.
//

#include "file_input_indexing.h"

#include <string>
#include <fstream>

#include "file_loader.h"
#include "util/pretty_print.h"
#include "util/primitives/parasort_cmp.h"

#define TLS_WRITE_BUFF_SIZE (1024 * 1024)
//#define GetIndexArr GetMallocPReadArrReadOnly
#define GetIndexArr GetMMAPArrReadOnly
using namespace std;

void FileInputHelper::ParseCustomerInputFile(const char *customer_path) {
    int32_t max_id = INT32_MIN;
    int32_t min_id = INT32_MAX;
    auto customers = (Customer *) malloc(sizeof(Customer) * MAX_NUM_CUSTOMERS);
    volatile uint32_t size_of_customers = 0;
    {
        FileLoader loader(customer_path);
#pragma omp parallel num_threads(io_threads)
        {
            uint32_t buf_cap = TLS_WRITE_BUFF_SIZE;
            auto *local_buffer = (Customer *) malloc(sizeof(Customer) * buf_cap);
            ConsumerBuffer customer_write_buffer(local_buffer, buf_cap, customers, &size_of_customers);
            char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for reduction(max:max_id), reduction(min:min_id)
            for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
                ssize_t num_reads = loader.ReadToBuf(i, tmp);
                ParseCustomer({.buf_= tmp, .size_= num_reads}, lock_free_linear_table_,
                              customer_write_buffer, max_id, min_id);
            }
            customer_write_buffer.submit_if_possible();
#pragma omp barrier
            // may have uninitialized garbage slots, but it is okay.
#pragma omp single
            {
                customer_categories_ = (int32_t *) malloc(sizeof(int32_t) * (max_id + 1));
            }
#pragma omp for
            for (size_t i = 0; i < size_of_customers; i++) {
                Customer customer = customers[i];
                customer_categories_[customer.key] = customer.category;
            }
            free(tmp);
            free(local_buffer);
        }
#ifdef DEBUG
        log_info("Beg ... [1, 11] customer categories...");
        for (auto i = 1; i < 11; i++)
            lock_free_linear_table_.PrintSlot(customer_categories_[i]);
        log_info("End ... [1, 11] customer categories...");
#endif
        log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_customers, min_id, max_id,
                 max_id - min_id + 1);
        loader.PrintEndStat();
    }
    // Free Customers.
    free(customers);
    lock_free_linear_table_.PrintTable();
}

void FileInputHelper::ParseOrderInputFile(const char *order_path) {
    auto orders = (Order *) malloc(sizeof(Order) * MAX_NUM_ORDERS);
    reordered_orders_ = (Order *) malloc(sizeof(Order) * MAX_NUM_ORDERS);
    vector<uint32_t> histogram;
    uint32_t *cur_write_off = nullptr;
    uint32_t max_order_date = 0;
    uint32_t min_order_date = UINT32_MAX;
    Timer timer;
    {
#ifdef FILE_LOAD_PREAD
        FileLoader loader(order_path);
#else
        FileLoaderMMap loader(order_path);
#endif
#pragma omp parallel num_threads(io_threads)
        {
            uint32_t buf_cap = TLS_WRITE_BUFF_SIZE;
            auto *local_buffer = (Order *) malloc(sizeof(Order) * buf_cap);
            OrderBuffer order_write_buffer(local_buffer, buf_cap, orders, &size_of_orders);
            char *local_buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for reduction(max:max_order_date) reduction(min:min_order_date)
            for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
#ifdef FILE_LOAD_PREAD
                ssize_t num_reads = loader.ReadToBuf(i, local_buf);
                ParseOrder({.buf_= local_buf, .size_= num_reads}, order_write_buffer,
                           max_order_date, min_order_date);
#else
                char *tmp = nullptr;
                ssize_t num_reads = loader.ReadToBuf(i, tmp, local_buf);
                ParseOrder({.buf_= tmp, .size_= num_reads}, order_write_buffer,
                           max_order_date, min_order_date);
#endif
            }
            order_write_buffer.submit_if_possible();

#pragma omp barrier
#pragma omp single
            {
                timer.reset();
            }
#pragma omp for
            for (size_t i = 0; i < size_of_orders; i++) {
                auto &order = orders[i];
                order.customer_key = customer_categories_[order.customer_key];
            }
#pragma omp single
            {
                free(customer_categories_);
            }
            int32_t second_level_range = max_order_date - min_order_date + 1;
            int32_t num_buckets = second_level_range * lock_free_linear_table_.Size();
            BucketSortSmallBuckets(histogram, orders, reordered_orders_, cur_write_off, order_bucket_ptrs_,
                                   size_of_orders, num_buckets,
                                   [orders, min_order_date, second_level_range](uint32_t it) {
                                       Order order = orders[it];
                                       return order.customer_key * second_level_range
                                              + order.order_date_bucket - min_order_date;
                                   }, io_threads, &timer);
            assert(order_bucket_ptrs_[num_buckets] == MAX_NUM_ORDERS);
            free(local_buf);
            free(local_buffer);
        }
        max_order_date_ = max_order_date;
        min_order_date_ = min_order_date;
        log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_orders, min_order_date_,
                 max_order_date_, max_order_date_ - min_order_date_ + 1);
        loader.PrintEndStat();
    }
    free(cur_write_off);
    free(orders);
}

void FileInputHelper::WriteOrderIndexToFIle(const char *order_path) {
    // Free Orders, Saving Order Index.
    Timer write_timer;
    string prefix = order_path;
    string order_key_path = prefix + ORDER_KEY_BIN_FILE_SUFFIX;
    string order_date_path = prefix + ORDER_DATE_BIN_FILE_SUFFIX;
    string order_meta_path = prefix + ORDER_META_BIN_FILE_SUFFIX;
    {
        ofstream ofs(order_meta_path, std::ofstream::out);
        Archive<ofstream> ar(ofs);
        auto category_table = lock_free_linear_table_.GetTable();
        for (auto e: category_table) {
            e.PrintStr();
        }
        const char *test_chars = "BUILDING";
        log_info("Probe Test: %d", LinearProbe(category_table, test_chars, 0, strlen(test_chars)));

        int32_t second_level_range = max_order_date_ - min_order_date_ + 1;
        int32_t num_buckets = second_level_range * lock_free_linear_table_.Size();
        vector<uint32_t> bucket_ptrs_stl(num_buckets + 1);
        memcpy(&bucket_ptrs_stl.front(), order_bucket_ptrs_, (num_buckets + 1) * sizeof(uint32_t));
        ar << category_table << min_order_date_ << max_order_date_
           << second_level_range << num_buckets << bucket_ptrs_stl;
        ofs.flush();
    }
    log_info("Meta Cost: %.6lfs", write_timer.elapsed());
    // Key, Date...
    int fd;
    auto *key_output = GetMMAPArr<int32_t>(order_key_path.c_str(), fd, size_of_orders);
    auto *date_output = GetMMAPArr<uint32_t>(order_date_path.c_str(), fd, size_of_orders);
#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < size_of_orders; i++) {
            key_output[i] = reordered_orders_[i].key;
            date_output[i] = reordered_orders_[i].order_date_bucket;
        }
    }
    log_info("Write Time: %.9lfs, TPS: %.3lf G/s", write_timer.elapsed(),
             sizeof(int32_t) * (size_of_orders) * 2 / write_timer.elapsed() / pow(10, 9));
#ifdef M_UNMAP
    munmap(key_output, size_of_orders * sizeof(int32_t));
    munmap(date_output, size_of_orders * sizeof(uint32_t));
#endif
    free(reordered_orders_);
}

void FileInputHelper::ParseLineItemInputFile(const char *line_item_path) {
    uint32_t max_ship_date = 0;
    uint32_t min_ship_date = UINT32_MAX;
    auto items = (LineItem *) malloc(sizeof(LineItem) * MAX_NUM_ITEMS);
    reordered_items_ = (LineItem *) malloc(sizeof(LineItem) * MAX_NUM_ITEMS);
    uint32_t *cur_write_off_item = nullptr;
    Timer timer;
    vector<uint32_t> histogram;
    {
#ifdef FILE_LOAD_PREAD
        FileLoader loader(line_item_path);
#else
        FileLoaderMMap loader(line_item_path);
#endif
#pragma omp parallel num_threads(io_threads)
        {
            uint32_t buf_cap = TLS_WRITE_BUFF_SIZE;
            auto *local_buffer = (LineItem *) malloc(sizeof(LineItem) * buf_cap);
            LineItemBuffer item_write_buffer(local_buffer, buf_cap, items, &size_of_items_);
            char *local_buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for reduction(max:max_ship_date) reduction(min:min_ship_date)
            for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
#ifdef FILE_LOAD_PREAD
                ssize_t num_reads = loader.ReadToBuf(i, local_buf);
                ParseLineItem({.buf_= local_buf, .size_= num_reads}, item_write_buffer,
                              max_ship_date, min_ship_date);
#else
                char *tmp = nullptr;
                    ssize_t num_reads = loader.ReadToBuf(i, tmp, local_buf);
                    ParseLineItem({.buf_= tmp, .size_= num_reads}, item_write_buffer,
                                  max_ship_date, min_ship_date);
#endif
            }
            item_write_buffer.submit_if_possible();

#pragma omp barrier
#pragma omp single
            {
                timer.reset();
            }
            int32_t num_buckets = max_ship_date - min_ship_date + 1;
            BucketSortSmallBuckets(histogram, items, reordered_items_,
                                   cur_write_off_item, bucket_ptrs_item_, size_of_items_, num_buckets,
                                   [items, min_ship_date](uint32_t it) {
                                       return items[it].ship_date_bucket - min_ship_date;
                                   }, io_threads, &timer);
            free(local_buf);
            free(local_buffer);
        }
        max_ship_date_ = max_ship_date;
        min_ship_date_ = min_ship_date;
        log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_items_, min_ship_date,
                 max_ship_date_, max_ship_date_ - min_ship_date_ + 1);
        loader.PrintEndStat();
    }
    free(cur_write_off_item);
    free(items);
}

void FileInputHelper::WriteLineItemIndexToFile(const char *line_item_path) {
    Timer write_timer;
    int fd;
    string prefix = line_item_path;
    string item_order_id_path = prefix + LINE_ITEM_ORDER_KEY_FILE_SUFFIX;
    string item_price_path = prefix + LINE_ITEM_PRICE_FILE_SUFFIX;
    string item_meta_path = prefix + LINE_ITEM_META_BIN_FILE_SUFFIX;
    {
        ofstream ofs(item_meta_path, std::ofstream::out);
        Archive<ofstream> ar(ofs);
        int32_t num_buckets = max_ship_date_ - min_ship_date_ + 1;
        vector<uint32_t> bucket_ptrs_stl(num_buckets + 1);
        memcpy(&bucket_ptrs_stl.front(), bucket_ptrs_item_, sizeof(uint32_t) * (num_buckets + 1));
        ar << min_ship_date_ << max_ship_date_ << num_buckets << bucket_ptrs_stl;
        ofs.flush();
    }
    log_info("Meta Cost: %.6lfs", write_timer.elapsed());
    // Key, Date...
    auto *lt_order_id_output = GetMMAPArr<int32_t>(item_order_id_path.c_str(), fd, size_of_items_);
    auto *lt_price_output = GetMMAPArr<double>(item_price_path.c_str(), fd, size_of_items_);
#pragma omp parallel
    {
#pragma omp for
        for (size_t i = 0; i < size_of_items_; i++) {
            lt_order_id_output[i] = reordered_items_[i].order_key;
            lt_price_output[i] = reordered_items_[i].price;
        }
    }
#ifdef M_UNMAP
    munmap(lt_order_id_output, size_of_items_ * sizeof(int32_t));
    munmap(lt_price_output, size_of_items_ * sizeof(double));
#endif
    free(reordered_items_);
    log_info("Write Time: %.9lfs, TPS: %.3lf G/s", write_timer.elapsed(),
             (sizeof(int32_t) + sizeof(double)) * (size_of_items_) / write_timer.elapsed() / pow(10, 9));
}


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
    order_dates_ = GetIndexArr<uint32_t>(order_date_path.c_str(), fd, size_of_orders_);
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

struct Result {
    uint32_t order_offset;
    double price;
};

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
    auto acc_prices = (double *) malloc(sizeof(double) * order_array_view_size);
    auto relative_off = (uint32_t *) malloc(sizeof(uint32_t) * order_array_view_size);
    auto results = (Result *) malloc(sizeof(Result) * order_array_view_size);
    uint32_t *order_pos_dict;
    bool *bmp;
    int32_t size_of_results = 0;
    log_info("%d", order_keys_[order_bucket_ptrs_[o_bucket_beg]]);
    int32_t max_order_id = 0;
    vector<uint32_t> histogram;

    Timer timer;
    auto order_bucket_ptr_beg = order_bucket_ptrs_[o_bucket_beg];
    auto order_bucket_ptr_end = order_bucket_ptrs_[o_bucket_end];
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
        for (size_t i = item_bucket_ptrs_[item_bucket_beg]; i < item_bucket_ptrs_[item_bucket_end]; i++) {
            auto order_key = item_order_keys_[i];
            if (bmp[order_key]) {
                acc_prices[order_pos_dict[order_key]] += item_prices_[i];
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
    printf("l_orderkey|o_orderdate|revenue\n");
    for (auto i = 0; i < min<int32_t>(size_of_results, limit); i++) {
        char date[DATE_LEN + 1];
        date[DATE_LEN] = '\0';
        ConvertBucketIDToDate(date, order_dates_[results[i].order_offset]);
        printf("%d|%s|%.2lf\n",
               order_keys_[results[i].order_offset], date, results[i].price);
    }
    log_info("Query Time: %.6lfs", timer.elapsed());
}
