//
// Created by yche on 10/10/19.
//
#include <unordered_map>

#include "util/program_options/popl.h"
#include "file_loader.h"
#include "file_input_indexing.h"

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

        auto io_threads = omp_get_max_threads();
        log_info("Num Threads: %d", io_threads);
        // 1st: Init Customer List.
        FileInputHelper file_input_helper(io_threads);
        file_input_helper.ParseCustomerInputFile(customer_path);

        // 2nd: Init Order List.
        file_input_helper.ParseOrderInputFile(order_path);
        file_input_helper.WriteOrderIndexToFIle(order_path);

        // 3rd: Init LineItem List.
        file_input_helper.ParseLineItemInputFile(line_item_path);
        file_input_helper.WriteLineItemIndexToFile(line_item_path);

        IndexHelper index_helper(order_path, line_item_path);
        // Query Top K, Given Three Filters.
        if (customer_filter_option->is_set() && order_filter_option->is_set() && line_item_filter_option->is_set()
            && limit_option->is_set()) {
            string category = customer_filter_option.get()->value();
            string order_date = order_filter_option.get()->value();
            string ship_date = line_item_filter_option.get()->value();
            int limit = limit_option.get()->value();
            log_info("Filter: %s, %s, %s; Limit: %d", category.c_str(), order_date.c_str(), ship_date.c_str(),
                     limit);
#ifdef DEBUG
            const char *date = order_filter_option.get()->value().c_str();
            char date2[DATE_LEN];
            log_info("%d, %d", ConvertDateToBucketID(order_date.c_str()), ConvertDateToBucketID(ship_date.c_str()));
            ConvertBucketIDToDate(date2, ConvertDateToBucketID(date));
            log_info("%.*s", DATE_LEN, date2);
#endif
            index_helper.Query(category, order_date, ship_date, limit);
        }
    }
    log_info("Mem Usage: %d KB", getValue());
}