#pragma once

#include <cmath>

#include <thread>

#include <omp.h>

#include "util/primitives/blockingconcurrentqueue.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/primitives/primitives.h"
#include "util/primitives/blocking_queue.h"

#define IO_REQ_SIZE (4 * 1024 * 1024)
#define EXTRA_IO_SIZE (4 * 1024)

#define NUM_IO_THREADS (40)
#define NUM_PARSERS (16)
#define NUM_BUFFERS (NUM_IO_THREADS +NUM_PARSERS)    // Must >= IO_THREADS

#define LINUX_SPLITTER ('\n')
#define COL_SPLITTER ('|')

#define CUSTOMER_CATEGORY_LEN (10)
#define DATE_LEN (10)

struct ParsingTask {
    char *buf_;
    ssize_t size_;
};

inline size_t FindStartIdx(char *buf) {
    size_t i = EXTRA_IO_SIZE;
    while (buf[i - 1] != LINUX_SPLITTER) {
        i--;
    }
    return i;
}

inline size_t LinearSearch(const char *str, size_t i, size_t len, char token) {
    while (i < len && str[i] != token) { i++; }
    return i;
}

inline int32_t StrToInt(const char *str, size_t beg, size_t end) {
    int sum = str[beg] - '0';
    for (auto i = beg + 1; i < end; i++) {
        sum = sum * 10 + (str[i] - '0');
    }
    return sum;
}

double latter_digits[] = {
        1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
};

inline double StrToFloat(const char *p, size_t beg, size_t end) {
    double r = 0.0;
    // Assume already 0-9 chars.
    for (; beg < end && p[beg] != '.'; beg++) {
        r = (r * 10.0) + (p[beg] - '0');
    }
    assert(p[beg] == '.');
    beg++;

    double frac = 0.0;
    auto frac_size = end - beg;
    for (; beg < end; beg++) {
        frac = (frac * 10.0) + (p[beg] - '0');
    }
    r += frac * latter_digits[frac_size];
    return r;
}

#define Y_MUL (10000)
#define M_MUL (100)

// Assume YYYY-MM-DD
inline uint32_t ConvertDateToUint32(const char *date) {
    char buf[11];
    memcpy(buf, date, sizeof(char) * 11);
    buf[4] = '\0';
    buf[7] = '\0';
    buf[10] = '\0';
    return Y_MUL * StrToInt(buf, 0, 4) + M_MUL * StrToInt(buf, 5, 7) + StrToInt(buf, 8, 10);
}

// Asssume Large Enough for "YYYY-MM-DD" (10 chars)
inline void ConvertUint32ToDate(char *date, uint32_t val) {
    stringstream ss;
    ss << std::setw(4) << std::setfill('0') << val / Y_MUL << "-";
    val %= Y_MUL;
    ss << std::setw(2) << val / M_MUL << "-";
    val %= M_MUL;
    ss << std::setw(2) << val;
    memcpy(date, ss.str().c_str(), 10);
}


inline size_t ParseConsumer(ParsingTask task, char *strs, atomic_int &counter, mutex &mtx) {
    auto buf = task.buf_;
    auto i = FindStartIdx(buf);
    size_t size = 0;

    while (i < task.size_) {
        // 1st: CID.
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;

        int32_t id = StrToInt(buf, i, end);
        i = end + 1;

        // 2nd: Parse Category
        end = LinearSearch(buf, i, task.size_, LINUX_SPLITTER);
        if (end == task.size_)return size;

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
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t id = StrToInt(buf, i, end);
        i = end + 1;
        assert(id > 0);

        // 2nd: CID.
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t cid = StrToInt(buf, i, end);
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
        size_t end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        int32_t id = StrToInt(buf, i, end);
        i = end + 1;
//        assert(id > 0);

        // 2nd: Price.
        end = LinearSearch(buf, i, task.size_, COL_SPLITTER);
        if (end == task.size_)return size;
        double price = StrToFloat(buf, i, end);
        i = end + 1;
//        assert(price > 0);

        // 3rd: Ship-Date.
        end = i + DATE_LEN;
        if (end >= task.size_)return size;
        uint32_t ship_date = ConvertDateToUint32(buf + i);
        i = end + 1;
        size++;
//        assert(ship_date > 0);
    }
    return size;
}

template<typename F>
void ParseFileSeq(const char *file_name, F f) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);

    size_t num_rows = 0;
    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
    log_info("Start IO, Size: %zu", size);
    for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
        ssize_t num_reads;
        if (i != 0) {
            num_reads = pread(file_fd, buf, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
        } else {
            num_reads = pread(file_fd, buf + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i) + EXTRA_IO_SIZE;
            buf[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
        }
        num_rows += f({.buf_= buf, .size_= num_reads});
    }
    free(buf);
    log_info("Finish IO, #Rows: %zu", num_rows);
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFilePipeLine(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);

    blocking_queue<ParsingTask> parsing_tasks;
    moodycamel::BlockingConcurrentQueue<char *> read_buffers;
    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char) * NUM_BUFFERS);
    for (auto i = 0; i < NUM_BUFFERS; i++) {
        read_buffers.enqueue(buf + i * IO_REQ_SIZE * sizeof(char));
    }
    vector<thread> threads(NUM_PARSERS);
    vector<size_t> tls_sum(NUM_PARSERS * CACHE_LINE_ENTRY);
    vector<size_t> tls_dequeue(NUM_PARSERS * CACHE_LINE_ENTRY);
    for (auto i = 0; i < NUM_PARSERS; i++) {
        threads[i] = thread([&read_buffers, &parsing_tasks, f, i, &tls_sum, &tls_dequeue]() {
            while (true) {
                ParsingTask task{.buf_=nullptr, .size_=0};
                task = parsing_tasks.pop();
                if (task.buf_ == nullptr) {
                    return;
                }
                // Consume the Buffer.
                tls_sum[i * CACHE_LINE_ENTRY] += f(task);
                tls_dequeue[i * CACHE_LINE_ENTRY]++;
                read_buffers.enqueue(task.buf_);
            }
        });
    }
    log_info("Start IO, Size: %zu", size);

    size_t num_enqueue = 0;
#pragma omp parallel num_threads(io_threads) reduction(+:num_enqueue)
    {
#pragma omp for
        for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            char *tmp = nullptr;
            ssize_t num_reads;
            read_buffers.try_dequeue(tmp);
            if (tmp == nullptr) {
                read_buffers.wait_dequeue(tmp);
            }
            if (i != 0) {
                num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
            } else {
                tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
                num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i)
                            + EXTRA_IO_SIZE;
            }
            assert(tmp != nullptr);
            parsing_tasks.push({.buf_= tmp, .size_= num_reads});
            num_enqueue++;
        }
    }
    for (auto i = 0; i < NUM_PARSERS; i++) {
        parsing_tasks.push({.buf_=nullptr, .size_=-1});
    }
    for (auto i = 0; i < NUM_PARSERS; i++) {
        threads[i].join();
    }
    free(buf);

    size_t num_rows = 0;
    size_t num_dequeue = 0;
    for (auto i = 0; i < NUM_PARSERS; i++) {
        num_rows += tls_sum[i * CACHE_LINE_ENTRY];
        num_dequeue += tls_dequeue[i * CACHE_LINE_ENTRY];
    }
    log_info("Finish IO, #Rows: %zu, #Enqueue: %zu, #Dequeue: %zu", num_rows, num_enqueue, num_dequeue);
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFilePRead(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);

    size_t num_rows = 0;
    log_info("Start IO, Size: %zu", size);
#pragma omp parallel num_threads(io_threads) reduction(+:num_rows)
    {
        char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));

#pragma omp for
        for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            ssize_t num_reads;

            if (i != 0) {
                num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
            } else {
                tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
                num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i)
                            + EXTRA_IO_SIZE;
            }
            num_rows += f({.buf_= tmp, .size_= num_reads});
        }
        free(tmp);
    }
    log_info("Finish IO, #Rows: %zu", num_rows);
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFileMMAP(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);
    char *mmap_mem = (char *) mmap(0, size, PROT_READ, MAP_PRIVATE, file_fd, 0);

    log_info("Start IO, Size: %zu", size);
    size_t num_rows = 0;
#pragma omp parallel num_threads(io_threads) reduction(+:num_rows)
    {
        char *first_buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
        char *tmp = nullptr;
#pragma omp for
        for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            ssize_t num_reads;
            if (i != 0) {
                num_reads = min<ssize_t>(size - i + EXTRA_IO_SIZE, IO_REQ_SIZE);
                tmp = mmap_mem + i - EXTRA_IO_SIZE;
            } else {
                num_reads = min<ssize_t>(size - i, IO_REQ_SIZE - EXTRA_IO_SIZE);
                memcpy(first_buf + EXTRA_IO_SIZE, mmap_mem, num_reads);
                tmp = first_buf;
                tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
                num_reads += EXTRA_IO_SIZE;
            }
            num_rows += f({.buf_= tmp, .size_= num_reads});
        }
        free(first_buf);
    }
    log_info("Finish IO, #Rows: %zu", num_rows);
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}