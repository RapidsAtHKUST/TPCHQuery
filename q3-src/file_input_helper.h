#pragma once

#include "lock_free_table.h"
#include "file_parser.h"

class FileInputHelper {
    // Environment.
    size_t io_threads;

    // Customer.
    int32_t *customer_categories_ = nullptr;
    LockFreeLinearTable lock_free_linear_table_;

    // Order.
    uint32_t max_order_date_ = 0;
    uint32_t min_order_date_ = UINT32_MAX;
    Order *reordered_orders_ = nullptr;
    volatile uint32_t size_of_orders = 0;
    uint32_t *order_bucket_ptrs_ = nullptr;

    // LineItems.
    uint32_t max_ship_date_ = 0;
    uint32_t min_ship_date_ = UINT32_MAX;
    LineItem *reordered_items_ = nullptr;
    volatile uint32_t size_of_items_ = 0;
    uint32_t *bucket_ptrs_item_ = nullptr;
public:
    explicit FileInputHelper(size_t io_threads) : io_threads(io_threads), lock_free_linear_table_(1024) {}

    void ParseCustomerInputFile(const char *customer_path);

    void ParseOrderInputFile(const char *order_path);

    void WriteOrderIndexToFIle(const char *order_path);

    void ParseLineItemInputFile(const char *line_item_path);

    void WriteLineItemIndexToFile(const char *line_item_path);
};