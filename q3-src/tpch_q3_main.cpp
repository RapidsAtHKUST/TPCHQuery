//
// Created by yche on 10/10/19.
//
#include <unordered_map>

#include "util/program_options/popl.h"
#include "file_parser.h"
#include "file_loader.h"

using namespace std;
using namespace popl;
using namespace chrono;

#define TLS_WRITE_BUFF_SIZE (1024 * 1024)

int main(int argc, char *argv[]) {
    log_info("%.9lf", StrToFloat("1.3", 0, 3));
    OptionParser op("Allowed options");
    auto customer_option = op.add<Value<string>>("c", "customer-path", "the customer file path");
    auto order_option = op.add<Value<string>>("o", "order-path", "the order file path");
    auto line_item_option = op.add<Value<string>>("l", "file-path", "the line item file path");

    auto customer_filter_option = op.add<Value<string>>("0", "cf", "the customer filter");
    auto order_filter_option = op.add<Value<string>>("1", "of", "the order filter");
    auto line_item_filter_option = op.add<Value<string>>("2", "lf", "the line item filter");
    auto limit_option = op.add<Value<int>>("3", "limit", "the limit number");
    op.parse(argc, argv);

    if (customer_option->is_set() && order_option->is_set() && line_item_option->is_set()) {
        log_info("Path: %s, %s, %s", customer_option.get()->value().c_str(),
                 order_option.get()->value().c_str(), line_item_option->value().c_str());
        // IO-Buffer, IO-Size in the Cap (max: 128KB)
        auto customer_path = customer_option.get()->value().c_str();
        auto order_path = order_option.get()->value().c_str();
        auto line_item_path = line_item_option.get()->value().c_str();

        auto io_threads = omp_get_max_threads();

        // 1st: Init Customer List.
        int32_t max_id = INT32_MIN;
        int32_t min_id = INT32_MAX;
        LockFreeLinearTable lock_free_linear_table(1024);
        auto customers = (Customer *) malloc(sizeof(Customer) * MAX_NUM_CUSTOMERS);
        int32_t *customer_categories;
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
                    ParseCustomer({.buf_= tmp, .size_= num_reads}, lock_free_linear_table,
                                  customer_write_buffer, max_id, min_id);
                }
                customer_write_buffer.submit_if_possible();
#pragma omp barrier
                // may have uninitialized garbage slots, but it is okay.
#pragma omp single
                {
                    customer_categories = (int32_t *) malloc(sizeof(int32_t) * (max_id + 1));
                }
#pragma omp for
                for (size_t i = 0; i < size_of_customers; i++) {
                    Customer customer = customers[i];
                    customer_categories[customer.key] = customer.category;
                }
                free(tmp);
                free(local_buffer);
            }
#ifdef DEBUG
            log_info("Beg ... [1, 11] customer categories...");
            for (auto i = 1; i < 11; i++)
                lock_free_linear_table.PrintSlot(customer_categories[i]);
            log_info("End ... [1, 11] customer categories...");
#endif
            log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_customers, min_id, max_id,
                     max_id - min_id + 1);
            loader.PrintEndStat();
        }


        lock_free_linear_table.PrintTable();

        // 2nd: Init Order List.
        uint32_t max_order_date = 0;
        uint32_t min_order_date = UINT32_MAX;
        auto orders = (Order *) malloc(sizeof(Order) * MAX_NUM_ORDERS);
        uint32_t *cur_write_off, *bucket_ptrs;
        auto reordered_orders = (Order *) malloc(sizeof(Order) * MAX_NUM_ORDERS);
        vector<uint32_t> histogram;

        volatile uint32_t size_of_orders = 0;
        Timer timer;
        {
            FileLoader loader(order_path);
#pragma omp parallel num_threads(io_threads)
            {
                uint32_t buf_cap = TLS_WRITE_BUFF_SIZE;
                auto *local_buffer = (Order *) malloc(sizeof(Order) * buf_cap);
                OrderBuffer order_write_buffer(local_buffer, buf_cap, orders, &size_of_orders);
                char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for reduction(max:max_order_date) reduction(min:min_order_date)
                for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
                    ssize_t num_reads = loader.ReadToBuf(i, tmp);
                    ParseOrder({.buf_= tmp, .size_= num_reads}, order_write_buffer,
                               max_order_date, min_order_date);
                }
                order_write_buffer.submit_if_possible();

#pragma omp barrier
#pragma omp single
                {
                    timer.reset();
                }
#pragma omp for
                for (auto i = 0; i < size_of_orders; i++) {
                    auto &order = orders[i];
                    order.customer_key = customer_categories[order.customer_key];
                }
                int32_t second_level_range = max_order_date - min_order_date + 1;
                int32_t num_buckets = second_level_range * lock_free_linear_table.Size();
                BucketSortSmallBuckets(histogram, orders, reordered_orders, cur_write_off, bucket_ptrs,
                                       size_of_orders, num_buckets,
                                       [orders, min_order_date, second_level_range](uint32_t it) {
                                           Order order = orders[it];
                                           return order.customer_key * second_level_range
                                                  + order.order_date_bucket - min_order_date;
                                       }, io_threads, &timer);
                free(tmp);
                free(local_buffer);
            }
            log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_orders, min_order_date,
                     max_order_date, max_order_date - min_order_date + 1);
            loader.PrintEndStat();
        }

        // 3rd: Init LineItem List.
        uint32_t max_ship_date = 0;
        uint32_t min_ship_date = UINT32_MAX;
        auto items = (LineItem *) malloc(sizeof(LineItem) * MAX_NUM_ITEMS);
        auto reordered_items = (LineItem *) malloc(sizeof(LineItem) * MAX_NUM_ITEMS);
        uint32_t *cur_write_off_item, *bucket_ptrs_item;
        volatile uint32_t size_of_items = 0;
        {
            FileLoader loader(line_item_path);
#pragma omp parallel num_threads(io_threads)
            {
                uint32_t buf_cap = TLS_WRITE_BUFF_SIZE;
                auto *local_buffer = (LineItem *) malloc(sizeof(LineItem) * buf_cap);
                LineItemBuffer item_write_buffer(local_buffer, buf_cap, items, &size_of_items);
                char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for reduction(max:max_ship_date) reduction(min:min_ship_date)
                for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
                    ssize_t num_reads = loader.ReadToBuf(i, tmp);
                    ParseLineItem({.buf_= tmp, .size_= num_reads}, item_write_buffer,
                                  max_ship_date, min_ship_date);
                }
                item_write_buffer.submit_if_possible();

#pragma omp barrier
#pragma omp single
                {
                    timer.reset();
                }
                int32_t num_buckets = max_ship_date - min_ship_date + 1;
                BucketSortSmallBuckets(histogram, items, reordered_items,
                                       cur_write_off_item, bucket_ptrs_item, size_of_items, num_buckets,
                                       [items, min_ship_date](uint32_t it) {
                                           return items[it].ship_date_bucket - min_ship_date;
                                       }, io_threads, &timer);
                free(tmp);
                free(local_buffer);
            }
            log_info("Finish IO, #Rows: %zu, Min-Max: [%d, %d], Range: %d", size_of_items, min_ship_date,
                     max_ship_date, max_ship_date - min_ship_date + 1);
            loader.PrintEndStat();
        }
    }
    log_info("Mem Usage: %d KB", getValue());

    if (customer_filter_option->is_set() && order_filter_option->is_set() && line_item_filter_option->is_set()) {
        log_info("Filter: %s, %s, %s", customer_filter_option.get()->value().c_str(),
                 order_filter_option.get()->value().c_str(), line_item_filter_option.get()->value().c_str());
        const char *date = order_filter_option.get()->value().c_str();
        char date2[11];
        date2[10] = '\0';

        log_info("%d", ConvertDateToBucketID(date));
        ConvertUint32ToDate(date2, ConvertDateToUint32(date));
        log_info("%s", date2);
    }
    if (limit_option->is_set()) {
        log_info("Limit: %d", limit_option.get()->value());
    }
}