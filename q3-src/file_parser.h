#pragma once

#include <cmath>
#include <cassert>

#include <thread>
#include <atomic>
#include <algorithm>

#include <omp.h>

#include "util/log.h"
#include "parsing_util.h"
#include "lock_free_table.h"

#define DATE_LEN (10)

//#define NAIVE_PARSING
using namespace std;

struct Consumer {

};

inline size_t ParseConsumer(ParsingTask task, LockFreeLinearTable &table, int &max_id, int &min_id) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    size_t size = 0;

    while (i < task.size_) {
        // 1st: CID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        int32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return size;
#endif
        i = end + 1;
        assert(id > 0);
        max_id = max(max_id, id);
        min_id = min(min_id, id);

        // 2nd: Parse Category
        end = LinearSearch(buf, i, task.size_, LINUX_SPLITTER);
        if (end == task.size_)return size;
        table.Insert(buf, i, end);
        i = end + 1;
        size++;
    }
    return size;
}

inline size_t ParseOrder(ParsingTask task) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    size_t size = 0;

    while (i < task.size_) {
        // 1st: OID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        int32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return size;
#endif
        i = end + 1;
        assert(id > 0);

        // 2nd: CID.
#ifdef NAIVE_PARSING
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t cid = StrToInt(buf, i, end);
#else
        end = task.size_;
        int32_t cid = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return size;
#endif
        i = end + 1;
        assert(cid > 0);

        // 3rd: Order-Date.
        end = i + DATE_LEN;
        if (end >= task.size_)return size;
        uint32_t order_date = ConvertDateToUint32(buf + i);
        i = end + 1;
        size++;
        assert(order_date > 0);
    }
    return size;
}

inline size_t ParseLineItem(ParsingTask task) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    size_t size = 0;
    while (i < task.size_) {
        // 1st: OID.
#ifdef NAIVE_PARSING
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t id = StrToInt(buf, i, end);
#else
        size_t end = task.size_;
        int32_t id = StrToIntOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return size;
#endif
        i = end + 1;
        assert(id > 0);

        // 2nd: Price.
#ifdef NAIVE_PARSING
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        double price = StrToFloat(buf, i, end);
#else
        end = task.size_;
        double price = StrToFloatOnline(buf, i, end, COL_SPLITTER);
        if (end == task.size_)return size;
#endif
        i = end + 1;
        assert(price > 0);

        // 3rd: Ship-Date.
        end = i + DATE_LEN;
        if (end >= task.size_)return size;
        uint32_t ship_date = ConvertDateToUint32(buf + i);
        i = end + 1;
        size++;
        assert(ship_date > 0);
    }
    return size;
}
