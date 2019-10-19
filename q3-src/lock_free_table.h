#pragma once

#include <atomic>
#include <mutex>

#define CUSTOMER_CATEGORY_LEN (10)
#define INVALID (-1)

using namespace std;

class LockFreeLinearTable {
    atomic_int counter;
    char *strs;
    mutex mtx;
public:
    explicit LockFreeLinearTable(int cap) :
            counter(0), strs((char *) malloc(cap * CUSTOMER_CATEGORY_LEN * sizeof(char))) {
    }

    int Insert(char *buf, size_t buf_beg, size_t buf_end) {
        int old_table_size = counter;
        auto probe = LinearProbe(buf, buf_beg, buf_end, 0, old_table_size);
        if (probe == INVALID) {
            unique_lock<mutex> lock(mtx);
            int new_table_size = counter;
            probe = LinearProbe(buf, buf_beg, buf_end, old_table_size, new_table_size);
            if (probe == INVALID) {
                memcpy(strs + new_table_size * CUSTOMER_CATEGORY_LEN, buf + buf_beg, buf_end - buf_beg);
                log_info("%.*s", buf_end - buf_beg, strs + new_table_size * CUSTOMER_CATEGORY_LEN);
                probe = new_table_size;
                counter++;
            }
        }
        return probe;
    }

    int LinearProbe(char *buf, size_t buf_beg, size_t buf_end, size_t it_beg, size_t it_end) {
        for (auto probe = it_beg; probe < it_end; probe++) {
            // Probe.
            if (memcmp(buf + buf_beg, strs + CUSTOMER_CATEGORY_LEN * probe, buf_end - buf_beg) == 0) {
                return probe;
            }
        }
        return INVALID;
    }
};