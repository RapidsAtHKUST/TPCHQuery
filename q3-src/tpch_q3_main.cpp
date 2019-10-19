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


        auto io_threads = std::thread::hardware_concurrency();
        int32_t max_id = INT32_MIN;
        int32_t min_id = INT32_MAX;
        LockFreeLinearTable lock_free_linear_table(1024);
        {
            FileLoader loader(customer_path);
            size_t num_rows = 0;
#pragma omp parallel num_threads(io_threads) reduction(+:num_rows) reduction(max:max_id), reduction(min:min_id)
            {
                char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for
                for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
                    ssize_t num_reads = loader.ReadToBuf(i, tmp);
                    num_rows += ParseConsumer({.buf_= tmp, .size_= num_reads}, lock_free_linear_table,
                                              max_id, min_id);
                }
                free(tmp);
            }
            log_info("Finish IO, #Rows: %zu, Min-Max: %d, %d", num_rows, min_id, max_id);
            loader.PrintEndStat();
        }
        lock_free_linear_table.PrintTable();
        ParseFilePRead(order_path, ParseOrder);
        ParseFilePRead(line_item_path, ParseLineItem);
    }
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