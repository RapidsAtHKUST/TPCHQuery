//
// Created by yche on 10/10/19.
//

#include <cassert>

#include <chrono>
#include <thread>

#include <omp.h>

#include "util/log.h"
#include "util/program_options/popl.h"
#include "util/primitives/parasort_cmp.h"
#include "util/timer.h"
#include "util/util.h"
#include "util/pretty_print.h"
#include "util/primitives/primitives.h"
#include "util/primitives/libpopcnt.h"
#include "util/primitives/boolarray.h"
#include "util/primitives/blockingconcurrentqueue.h"
#include "util/primitives/barrier.h"
#include "util/aio.h"

using namespace std;
using namespace popl;
using namespace std::chrono;

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

    Timer global_timer;
    if (customer_option->is_set() && order_option->is_set() && line_item_option->is_set()) {
        log_info("Path: %s, %s, %s", customer_option.get()->value().c_str(),
                 order_option.get()->value().c_str(), line_item_option->value().c_str());
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