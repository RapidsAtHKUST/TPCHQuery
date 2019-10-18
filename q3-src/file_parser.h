#pragma once

#include <cmath>

#include <thread>

#include <omp.h>

#include "util/primitives/blockingconcurrentqueue.h"
#include "util/log.h"
#include "util/timer.h"

#define IO_REQ_SIZE (128 * 1024)
#define EXTRA_IO_SIZE (4 * 1024)

#define NUM_IO_THREADS (16)
#define NUM_PARSERS (16)
#define NUM_BUFFERS (NUM_IO_THREADS +NUM_PARSERS)    // Must >= IO_THREADS

#define LINUX_SPLITTER ('\n')
#define COL_SPLITTER ('|')

#define CUSTOMER_CATEGORY_LEN (10)

struct ParsingTask {
    char *buf_;
    ssize_t size_;
};

inline size_t FindStartIdx(char* buf){
    size_t i = EXTRA_IO_SIZE;
    while (buf[i - 1] != LINUX_SPLITTER) {
        i--;
    }
    return  i;
}

inline size_t LinearSearch(const char *str, size_t i, size_t len, char token) {
    while (i < len && str[i] != token) { i++; }
    return i;
}

inline int32_t atoi_range(const char *str, size_t beg, size_t end) {
    assert(beg < end);
    int sum = str[beg] - '0';
    for (auto i = beg + 1; i < end; i++) {
        sum = sum * 10 + (str[i] - '0');
    }
    return sum;
}

template<typename F>
void ParseFileSeq(const char *file_name, F f) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);

    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
    log_info("Start IO");
    for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
        ssize_t num_reads;
        if (i != 0) {
            num_reads = pread(file_fd, buf, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
            assert(num_reads <= IO_REQ_SIZE);
        } else {
            num_reads = pread(file_fd, buf + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i);
            buf[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
            assert(num_reads <= IO_REQ_SIZE - EXTRA_IO_SIZE);
        }
        f({.buf_= buf, .size_= num_reads});
    }
    free(buf);
    log_info("Finish IO");
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

    moodycamel::BlockingConcurrentQueue<ParsingTask> parsing_tasks;
    moodycamel::BlockingConcurrentQueue<char *> read_buffers;
    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char) * NUM_BUFFERS);
    for (auto i = 0; i < NUM_BUFFERS; i++) {
        read_buffers.enqueue(buf + i * IO_REQ_SIZE * sizeof(char));
    }
    vector<thread> threads(NUM_PARSERS);
    for (auto i = 0; i < NUM_PARSERS; i++) {
        threads[i] = thread([&read_buffers, &parsing_tasks, f]() {
            while (true) {
                ParsingTask task{.buf_=nullptr, .size_=0};
                parsing_tasks.wait_dequeue(task);
                if (task.buf_ == nullptr) {
                    return;
                }
                // Consume the Buffer.
                f(task);
                read_buffers.enqueue(task.buf_);
            }
        });
    }
    log_info("Start IO");
#pragma omp parallel num_threads(io_threads)
    {
#pragma omp for
        for (size_t i = 0; i < size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            char *tmp = nullptr;
            ssize_t num_reads;
            read_buffers.try_dequeue(tmp);
            if (tmp == nullptr) {
                read_buffers.wait_dequeue(tmp);
            }
            assert(tmp != nullptr);
            if (i != 0) {
                num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
                assert(num_reads <= IO_REQ_SIZE);
            } else {
                tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
                num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i);
                assert(num_reads <= IO_REQ_SIZE - EXTRA_IO_SIZE);
            }
            parsing_tasks.enqueue({.buf_= tmp, .size_= num_reads});
        }
    }
    log_info("Finish IO");
    for (auto i = 0; i < NUM_PARSERS; i++) {
        parsing_tasks.enqueue({.buf_=nullptr, .size_=-1});
    }
    for (auto i = 0; i < NUM_PARSERS; i++) {
        threads[i].join();
    }
    free(buf);
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}