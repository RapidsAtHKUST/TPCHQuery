#pragma once

#define FILE_LOAD_PREAD

#define ORDER_KEY_BIN_FILE_SUFFIX ("_ORDER_KEY.rapids")
#define ORDER_DATE_BIN_FILE_SUFFIX ("_ORDER_DATE.rapids")
#define ORDER_META_BIN_FILE_SUFFIX ("_ORDER_META.rapids")

#define LINE_ITEM_ORDER_KEY_FILE_SUFFIX ("_LINE_ITEM_KEY.rapids")
#define LINE_ITEM_PRICE_FILE_SUFFIX ("_LINE_ITEM_PRICE.rapids")
#define LINE_ITEM_META_BIN_FILE_SUFFIX ("_LINE_ITEM_META.rapids")

#define MAX_NUM_CUSTOMERS (15000000)
#define MAX_NUM_ORDERS (150000000)
#define MAX_NUM_ITEMS (600037902)

#define DATE_LEN (10)

#define IO_REQ_SIZE (64 * 1024)
#define EXTRA_IO_SIZE (4 * 1024)

