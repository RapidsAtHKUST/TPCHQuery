//
// Created by yche on 10/10/19.
//

#include "util/program_options/popl.h"
#include "util/primitives/parasort_cmp.h"
#include "util/util.h"

#include "file_parser.h"

using namespace std;
using namespace popl;
using namespace std::chrono;

inline size_t LinearSearch(const char *str, size_t i, size_t len, char token) {
    while (i < len && str[i] != '&') { i++; }
    return i;
}

inline int atoi_range(const char *str, size_t beg, size_t end) {
    assert(beg < end);
    int sum = str[beg];
    for (auto i = beg + 1; i < end; i++) {
        sum = sum * 10 + str[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    OptionParser op("Allowed options");
    auto customer_option = op.add<Value<std::string>>("c", "customer-path", "the customer file path");
    auto order_option = op.add<Value<std::string>>("o", "order-path", "the order file path");
    auto line_item_option = op.add<Value<std::string>>("l", "file-path", "the line item file path");

    auto customer_filter_option = op.add<Value<std::string>>("0", "cf", "the custormer filter");
    auto order_filter_option = op.add<Value<std::string>>("1", "of", "the order filter");
    auto line_item_filter_option = op.add<Value<std::string>>("2", "lf", "the line item filter");
    auto limit_option = op.add<Value<int>>("3", "limit", "the limit number");
    op.parse(argc, argv);

    if (customer_option->is_set() && order_option->is_set() && line_item_option->is_set()) {
        log_info("Path: %s, %s, %s", customer_option.get()->value().c_str(),
                 order_option.get()->value().c_str(), line_item_option->value().c_str());
        //        auto customer_path = line_item_option.get()->value().c_str();

        // IO-Buffer, IO-Size in the Cap (max: 128KB)
        auto customer_path = customer_option.get()->value().c_str();
        ParseFile(customer_path, [](ParsingTask task) {
            size_t i = EXTRA_IO_SIZE - 1;
            auto buf = task.buf_;
            while (buf[i] != LINUX_SPLITTER) {
                i--;
                assert(i >= 0);
            }
            assert(buf[i] == LINUX_SPLITTER);
            while (i < task.size_) {
                // 1st: consumer ID.
                size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
                if (end == task.size_)
                    return;
                int id = atoi_range(buf, i, end);

                // 2nd: char [10]
                i = end + 1;
                if (end == task.size_)
                    return;
                
            }
        }, IO_THREADS);
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