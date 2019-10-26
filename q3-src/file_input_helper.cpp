//
// Created by yche on 10/23/19.
//

#include <fstream>

#include "file_input_helper.h"
#include "file_loader.h"

#include "config.h"

#define TLS_WRITE_BUFF_SIZE (1024 * 1024)

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
                                   }, &timer);
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
                                   }, &timer);
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