//
// Created by yche on 10/10/19.
//
#include <unordered_map>
#include <mutex>

#include "util/program_options/popl.h"
#include "util/util.h"
#include "file_parser.h"

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

        atomic_int counter(0);
        char *strs = (char *) malloc(1024 * CUSTOMER_CATEGORY_LEN * sizeof(char));
        mutex mtx;
        ParseFileSelf(customer_path, [&counter, &strs, &mtx](ParsingTask task) {
            ParseConsumer(task, strs, counter, mtx);
        });
        ParseFileSelf(order_path, [](ParsingTask task) {
            ParseOrder(task);
        });
        ParseFileSelf(line_item_path, [](ParsingTask task) {
            ParseLineItem(task);
        });
    }
    if (customer_filter_option->is_set() && order_filter_option->is_set() && line_item_filter_option->is_set()) {
        log_info("Filter: %s, %s, %s", customer_filter_option.get()->value().c_str(),
                 order_filter_option.get()->value().c_str(), line_item_filter_option.get()->value().c_str());
        const char *date = order_filter_option.get()->value().c_str();
        char date2[11];
        date2[10] = '\0';
        log_info("%d", ConvertDateToUint32(date));
        ConvertUint32ToDate(date2, ConvertDateToUint32(date));
        log_info("%s", date2);
    }
    if (limit_option->is_set()) {
        log_info("Limit: %d", limit_option.get()->value());
    }
}