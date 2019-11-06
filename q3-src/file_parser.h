#pragma once

#include <cmath>
#include <cassert>

#include <thread>
#include <atomic>
#include <algorithm>

#include <omp.h>

#include "config.h"
#include "util/log.h"
#include "util/primitives/local_buffer.h"
#include "parsing_util.h"
#include "lock_free_table.h"

//#define NAIVE_PARSING
using namespace std;

struct Customer {
    uint32_t key;
    int32_t category;
};

struct Order {
    uint32_t key;
    uint32_t customer_key;
    uint32_t order_date_bucket;
};

struct LineItem {
    uint32_t order_key;
    uint32_t ship_date_bucket;
    double price;
};

using ConsumerBuffer = LocalWriteBuffer<Customer, uint32_t>;
using OrderBuffer = LocalWriteBuffer<Order, uint32_t>;
using LineItemBuffer = LocalWriteBuffer<LineItem, uint32_t>;

inline void ParseCustomer(ParsingTask task, LockFreeLinearTable &table, ConsumerBuffer &local_write_buffer,
                          uint32_t &max_id, uint32_t &min_id) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);

    while (i < task.size_) {
        // 1st: CID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;
        uint32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        uint32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return;
#endif
        i = end + 1;
        assert(id > 0);
        max_id = max(max_id, id);
        min_id = min(min_id, id);

        // 2nd: Parse Category
        end = LinearSearch(buf, i, task.size_, LINUX_SPLITTER);
        if (end == task.size_)return;
        auto category = table.Insert(buf, i, end);
        assert(category != INVALID);
        i = end + 1;

        local_write_buffer.push({.key=id, .category=category});
    }
}

inline void ParseOrder(ParsingTask task, OrderBuffer &local_write_buffer,
                       uint32_t &max_date, uint32_t &min_date) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);

    while (i < task.size_) {
        // 1st: OID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;
        uint32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        uint32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return;
#endif
        i = end + 1;
        assert(id > 0);

        // 2nd: CID.
#ifdef NAIVE_PARSING
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;
        uint32_t cid = StrToInt(buf, i, end);
#else
        end = task.size_;
        uint32_t cid = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return;
#endif
        i = end + 1;
        assert(cid > 0);

        // 3rd: Order-Date.
        end = i + DATE_LEN;
        if (end >= task.size_)return;
        uint32_t order_date = ConvertDateToBucketID(buf + i);
        i = end + 1;
        max_date = max(max_date, order_date);
        min_date = min(min_date, order_date);
        assert(order_date > 0);
        local_write_buffer.push({.key=id, .customer_key=cid, .order_date_bucket = order_date});
    }
}

inline void ParseLineItem(ParsingTask task, LineItemBuffer &local_write_buffer,
                          uint32_t &max_date, uint32_t &min_date) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    while (i < task.size_) {
        // 1st: OID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;
        uint32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        uint32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return;
#endif
        i = end + 1;
        assert(id > 0);

        // 2nd: Price.
#ifdef NAIVE_PARSING
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return;
        double price = StrToFloat(buf, i, end);
#else
        end = task.size_;
        double price = StrToFloatOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return;
#endif
        i = end + 1;
        assert(price > 0);

        // 3rd: Ship-Date.
        end = i + DATE_LEN;
        if (end >= task.size_)return;
        uint32_t ship_date = ConvertDateToBucketID(buf + i);
        i = end + 1;
        assert(ship_date > 0);
        max_date = max(max_date, ship_date);
        min_date = min(min_date, ship_date);
        local_write_buffer.push({.order_key = id, .ship_date_bucket=ship_date, .price = price,});
    }
}
