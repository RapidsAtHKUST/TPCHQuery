#pragma once

#include <cstdint>

#include "lock_free_table.h"
#include "file_parser.h"

#define FILE_LOAD_PREAD

#define ORDER_KEY_BIN_FILE_SUFFIX ("_ORDER_KEY.rapids")
#define ORDER_DATE_BIN_FILE_SUFFIX ("_ORDER_DATE.rapids")
#define ORDER_META_BIN_FILE_SUFFIX ("_ORDER_META.rapids")

#define LINE_ITEM_ORDER_KEY_FILE_SUFFIX ("_LINE_ITEM_KEY.rapids")
#define LINE_ITEM_PRICE_FILE_SUFFIX ("_LINE_ITEM_PRICE.rapids")
#define LINE_ITEM_META_BIN_FILE_SUFFIX ("_LINE_ITEM_META.rapids")

using namespace std;

class FileInputHelper {
    size_t io_threads;
    int32_t *customer_categories;
    LockFreeLinearTable lock_free_linear_table;

public:
    explicit FileInputHelper(size_t io_threads) : io_threads(io_threads), customer_categories(nullptr),
                                                  lock_free_linear_table(1024) {
    }

    void ParseCustomerInputFile(const char *customer_path);

    void ParseOrderInputFile(const char *order_path);

    void ParseLineItemInputFile(const char *line_item_path);
};