//
// Created by yche on 10/10/19.
//
#include <unordered_map>
#include <mutex>

#include "util/program_options/popl.h"
#include "util/primitives/parasort_cmp.h"
#include "util/util.h"
#include "util/pretty_print.h"
#include "file_parser.h"

using namespace std;
using namespace popl;
using namespace chrono;

inline void ParseConsumer(ParsingTask task, char *strs, atomic_int &counter, mutex &mtx) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    auto prev_id = -1;
    while (i < task.size_) {
        // 1st: consumer ID.
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;

        // Parse ID.
        int32_t id = atoi_range(buf, i, end);
        assert(id > prev_id);
        prev_id = id;
        i = end + 1;

        // 2nd: char [10]
        end = LinearSearch(buf, i, task.size_, LINUX_SPLITTER);
        if (end == task.size_)return;

        // Parse Category.
        bool is_not_in = true;
        int old_table_size = counter;
        for (auto probe = 0; probe < old_table_size; probe++) {
            // Probe.
            if (memcmp(buf + i, strs + CUSTOMER_CATEGORY_LEN * probe, end - i) == 0) {
                is_not_in = false;
                break;
            }
        }
        // If Fail.
        if (is_not_in) {
            // Mutex Lock.
            unique_lock<mutex> lock(mtx);
            int new_table_size = counter;
            for (auto probe = old_table_size; probe < new_table_size; probe++) {
                if (memcmp(buf + i, strs + CUSTOMER_CATEGORY_LEN * probe, end - i) == 0) {
                    is_not_in = false;
                    break;
                }
            }
            if (is_not_in) {
                memcpy(strs + new_table_size * CUSTOMER_CATEGORY_LEN, buf + i, end - i);
                log_info("%.*s", end - i, strs + new_table_size * CUSTOMER_CATEGORY_LEN);
                counter++;
            }
        }
        i = end + 1;
    }
}

int main(int argc, char *argv[]) {
    OptionParser op("Allowed options");
    auto customer_option = op.add<Value<string>>("c", "customer-path", "the customer file path");
    auto order_option = op.add<Value<string>>("o", "order-path", "the order file path");
    auto line_item_option = op.add<Value<string>>("l", "file-path", "the line item file path");

    auto customer_filter_option = op.add<Value<string>>("0", "cf", "the custormer filter");
    auto order_filter_option = op.add<Value<string>>("1", "of", "the order filter");
    auto line_item_filter_option = op.add<Value<string>>("2", "lf", "the line item filter");
    auto limit_option = op.add<Value<int>>("3", "limit", "the limit number");
    op.parse(argc, argv);

    if (customer_option->is_set() && order_option->is_set() && line_item_option->is_set()) {
        log_info("Path: %s, %s, %s", customer_option.get()->value().c_str(),
                 order_option.get()->value().c_str(), line_item_option->value().c_str());
        // IO-Buffer, IO-Size in the Cap (max: 128KB)
        auto customer_path = customer_option.get()->value().c_str();
        atomic_int counter(0);

        char *strs = (char *) malloc(1024 * CUSTOMER_CATEGORY_LEN * sizeof(char));
        mutex mtx;
        ParseFilePipeLine(customer_path, [&counter, &strs, &mtx](ParsingTask task) {
            ParseConsumer(task, strs, counter, mtx);
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