#pragma once

#include <cmath>

#include <thread>

#include <omp.h>

#include "util/primitives/blockingconcurrentqueue.h"
#include "util/log.h"
#include "util/timer.h"

#define IO_REQ_SIZE (128 * 1024)
#define EXTRA_IO_SIZE (4 * 1024)
#define NUM_READ_BUFFERS (8)
#define NUM_PARSERS (8)
#define LINUX_SPLITTER ('\n')
#define COL_SPLITTER ('|')
#define IO_THREADS (8)

struct ParsingTask {
    char *buf_;
    ssize_t size_;
};

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFile(const char *file_name, F f, int io_threads = 1) {
    Timer timer;
    auto file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    auto size = file_size(file_name);

    moodycamel::BlockingConcurrentQueue<ParsingTask> parsing_tasks;
    moodycamel::BlockingConcurrentQueue<char *> read_buffers;
    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char) * NUM_READ_BUFFERS);
    for (auto i = 0; i < NUM_READ_BUFFERS; i++) {
        read_buffers.enqueue(buf + i * IO_REQ_SIZE);
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
            char *tmp;
            ssize_t num_reads;
            read_buffers.wait_dequeue(tmp);
            if (i != 0) {
                num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
            } else {
                tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
                num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i);
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
    log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
}