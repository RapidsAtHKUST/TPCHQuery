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
    assert(argc >= 5);
    string customer_path = argv[1];
    string order_path = argv[2];
    string line_item_path = argv[3];
    int num_queries = atoi(argv[4]);
    assert(argc >= num_queries * 4 + 5);

    auto io_threads = omp_get_max_threads();
    log_info("Num Threads: %d", io_threads);
    string order_key_path = order_path + ORDER_KEY_BIN_FILE_SUFFIX;
    string order_date_path = order_path + ORDER_DATE_BIN_FILE_SUFFIX;
    string order_meta_path = order_path + ORDER_META_BIN_FILE_SUFFIX;
    string item_order_id_path = line_item_path + LINE_ITEM_ORDER_KEY_FILE_SUFFIX;
    string item_price_path = line_item_path + LINE_ITEM_PRICE_FILE_SUFFIX;
    string item_meta_path = line_item_path + LINE_ITEM_META_BIN_FILE_SUFFIX;

    // 1st: Init Customer List.
    if (!file_exists(order_key_path) || !file_exists(order_date_path) || !file_exists(order_meta_path)
        || !file_exists(item_order_id_path) || !file_exists(item_price_path) || !file_exists(item_meta_path)) {
        FileInputHelper file_input_helper(io_threads);
        file_input_helper.ParseCustomerInputFile(customer_path.c_str());

        // 2nd: Init Order List.
        file_input_helper.ParseOrderInputFile(order_path.c_str());
        file_input_helper.WriteOrderIndexToFIle(order_path.c_str());

        // 3rd: Init LineItem List.
        file_input_helper.ParseLineItemInputFile(line_item_path.c_str());
        file_input_helper.WriteLineItemIndexToFile(line_item_path.c_str());
    }
    IndexHelper index_helper(order_path, line_item_path);
    for (auto i = 0; i < num_queries; i++) {
        string category = argv[i * 4 + 5 + 0];
        string order_date = argv[i * 4 + 5 + 1];
        string ship_date = argv[i * 4 + 5 + 2];
        int limit = atoi(argv[i * 4 + 5 + 3]);
        index_helper.Query(category, order_date, ship_date, limit);
    }
    log_info("Mem Usage: %d KB", getValue());
}