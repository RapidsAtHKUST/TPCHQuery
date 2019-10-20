#pragma once

#include <atomic>
#include <mutex>

#include "util/archive.h"

#define CUSTOMER_CATEGORY_LEN (10)
#define INVALID (-1)

using namespace std;

struct String {
    int16_t size;
    char chars[CUSTOMER_CATEGORY_LEN];

    void PrintStr() {
        log_info("%.*s", size, chars);
    }

    template<typename T>
    Archive<T> &Serialize(Archive<T> &archive) {
        archive & size & chars;
        return archive;
    }

    template<typename T>
    const Archive<T> &Serialize(const Archive<T> &archive) {
        archive & size & chars;
        return archive;
    }
};

int LinearProbe(vector<String> &strs, const char *buf, size_t buf_beg, size_t buf_end) {
    size_t it_beg = 0;
    size_t it_end = strs.size();
    for (auto probe = it_beg; probe < it_end; probe++) {
        // Probe.
        if (strs[probe].size == buf_end - buf_beg &&
            memcmp(buf + buf_beg, strs[probe].chars, buf_end - buf_beg) == 0) {
            return probe;
        }
    }
    assert(false);
    return INVALID;
}

class LockFreeLinearTable {
    atomic_int counter;
    vector<String> strs;
    mutex mtx;
public:
    explicit LockFreeLinearTable(int cap) :
            counter(0), strs(cap) {
    }

    int Size() {
        return counter;
    }

    vector<String> GetTable() {
        return vector<String>(begin(strs), begin(strs) + counter);
    }

    int Insert(char *buf, size_t buf_beg, size_t buf_end) {
        int old_table_size = counter;
        auto probe = LinearProbe(buf, buf_beg, buf_end, 0, old_table_size);
        if (probe == INVALID) {
            unique_lock<mutex> lock(mtx);
            int new_table_size = counter;
            probe = LinearProbe(buf, buf_beg, buf_end, old_table_size, new_table_size);
            if (probe == INVALID) {
                strs[new_table_size].size = buf_end - buf_beg;
                memcpy(strs[new_table_size].chars, buf + buf_beg, buf_end - buf_beg);
//                log_info("SizeOf: %zu", sizeof(String));
                log_info("%.*s", strs[new_table_size].size, strs[new_table_size].chars);
                probe = new_table_size;
                counter++;
            }
        }
        return probe;
    }

    int LinearProbe(char *buf, size_t buf_beg, size_t buf_end, size_t it_beg, size_t it_end) {
        for (auto probe = it_beg; probe < it_end; probe++) {
            // Probe.
            if (strs[probe].size == buf_end - buf_beg &&
                memcmp(buf + buf_beg, strs[probe].chars, buf_end - buf_beg) == 0) {
                return probe;
            }
        }
        return INVALID;
    }

    void PrintTable() {
        for (auto probe = 0; probe < counter; probe++) {
            strs[probe].PrintStr();
        }
    }

    void PrintSlot(int i) {
        strs[i].PrintStr();
    }
};