#pragma once

#include "util/primitives/primitives.h"
#include "util/primitives/blocking_queue.h"
#include "util/primitives/blockingconcurrentqueue.h"
#include "util/timer.h"
#include "util/util.h"
#include "parsing_util.h"

#define NUM_IO_THREADS (40)
#define NUM_PARSERS (16)
#define NUM_BUFFERS (NUM_IO_THREADS +NUM_PARSERS)    // Must >= IO_THREADS

template<typename T>
T *GetMMAPArr(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    auto ret = ftruncate(file_fd, file_size);

    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_WRITE, MAP_SHARED, file_fd, 0);
//    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_WRITE, MAP_SHARED | MAP_POPULATE, file_fd, 0);
    log_info("Open & MMAP Time: %.6lfs", populate_timer.elapsed());
    assert(ret == 0);
    return mmap_arr_;
}

template<typename T>
T *GetMMAPArrReadOnly(const char *file_name, int &file_fd, size_t arr_size) {
    Timer populate_timer;
    file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    size_t file_size = arr_size * sizeof(T);
    assert(file_fd >= 0);

    auto mmap_arr_ = (T *) mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);
    log_info("Open & MMAP Time: %.6lfs", populate_timer.elapsed());
    return mmap_arr_;
}

/*
 * Global One.
 */
class FileLoader {
protected:
    int file_fd{};
    Timer timer;
public:
    size_t size{};

    FileLoader() = default;

    explicit FileLoader(const char *file_name) {
        file_fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
        size = file_size(file_name);
        log_info("Start IO");
    }

    ssize_t ReadToBuf(size_t i, char *tmp) {
        ssize_t num_reads;
        if (i != 0) {
            num_reads = pread(file_fd, tmp, IO_REQ_SIZE, i - EXTRA_IO_SIZE);
        } else {
            tmp[EXTRA_IO_SIZE - 1] = LINUX_SPLITTER;
            num_reads = pread(file_fd, tmp + EXTRA_IO_SIZE, IO_REQ_SIZE - EXTRA_IO_SIZE, i)
                        + EXTRA_IO_SIZE;
        }
        return num_reads;
    }

    void PrintEndStat() {
        log_info("Read Time: %.6lfs, QPS: %.3lf GB/s", timer.elapsed(), size / timer.elapsed() / pow(10, 9));
    }
};

class FileLoaderMMap : public FileLoader {
    char *mmap_mem;
public:
    explicit FileLoaderMMap(const char *file_name) : FileLoader(file_name) {
        mmap_mem = (char *) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_fd, 0);
    }

    ssize_t ReadToBuf(size_t i, char *&tmp, char *first_buf) {
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
        return num_reads;
    }
};

template<typename F>
void ParseFileSeq(const char *file_name, F f) {
    size_t num_rows = 0;
    char *buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
    FileLoader loader(file_name);
    for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
        ssize_t num_reads = loader.ReadToBuf(i, buf);
        num_rows += f({.buf_= buf, .size_= num_reads});
    }
    free(buf);
    log_info("Finish IO, #Rows: %zu", num_rows);
    loader.PrintEndStat();
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFilePRead(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    FileLoader loader(file_name);
    size_t num_rows = 0;
#pragma omp parallel num_threads(io_threads) reduction(+:num_rows)
    {
        char *tmp = (char *) malloc(IO_REQ_SIZE * sizeof(char));
#pragma omp for
        for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            ssize_t num_reads = loader.ReadToBuf(i, tmp);
            num_rows += f({.buf_= tmp, .size_= num_reads});
        }
        free(tmp);
    }
    log_info("Finish IO, #Rows: %zu", num_rows);
    loader.PrintEndStat();
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFileMMAP(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    size_t num_rows = 0;
    FileLoaderMMap loader(file_name);
#pragma omp parallel num_threads(io_threads) reduction(+:num_rows)
    {
        char *first_buf = (char *) malloc(IO_REQ_SIZE * sizeof(char));
        char *tmp = nullptr;
#pragma omp for
        for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            ssize_t num_reads = loader.ReadToBuf(i, tmp, first_buf);
            num_rows += f({.buf_= tmp, .size_= num_reads});
        }
        free(first_buf);
    }
    log_info("Finish IO, #Rows: %zu", num_rows);
    loader.PrintEndStat();
}

/*
 * F requires a single parameter (ParsingTask)
 */
template<typename F>
void ParseFilePipeLine(const char *file_name, F f, int io_threads = NUM_IO_THREADS) {
    FileLoader loader(file_name);

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

    size_t num_enqueue = 0;
#pragma omp parallel num_threads(io_threads) reduction(+:num_enqueue)
    {
#pragma omp for
        for (size_t i = 0; i < loader.size; i += IO_REQ_SIZE - EXTRA_IO_SIZE) {
            char *tmp = nullptr;
            read_buffers.try_dequeue(tmp);
            if (tmp == nullptr) {
                read_buffers.wait_dequeue(tmp);
            }
            ssize_t num_reads = loader.ReadToBuf(i, tmp);
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
    loader.PrintEndStat();
}