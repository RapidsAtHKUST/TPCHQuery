#pragma once

#include <atomic>
#include <mutex>

#define CUSTOMER_CATEGORY_LEN (10)
#define INVALID (-1)

using namespace std;

struct String {
    int16_t size;
    char chars[CUSTOMER_CATEGORY_LEN];
};

class LockFreeLinearTable {
    atomic_int counter;
    String *strs;
    mutex mtx;
public:
    explicit LockFreeLinearTable(int cap) :
            counter(0), strs((String *) malloc(cap * sizeof(String))) {
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
};