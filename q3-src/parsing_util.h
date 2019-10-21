#pragma once

#include <cassert>

#include <iomanip>
#include <sstream>

#define IO_REQ_SIZE (64 * 1024)
#define EXTRA_IO_SIZE (4 * 1024)

#define LINUX_SPLITTER ('\n')
#define COL_SPLITTER ('|')

using namespace std;

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

inline int32_t StrToIntOnline(const char *str, size_t beg, size_t &end, char token) {
    if (beg == end) { return 0; }
    int sum = str[beg] - '0';
    auto i = beg + 1;
    for (; i < end && str[i] != token; i++) {
        sum = sum * 10 + (str[i] - '0');
    }
    end = i;
    return sum;
}

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
    static thread_local double latter_digits[] = {
            1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
    };
    r += frac * latter_digits[frac_size];
    return r;
}

inline double StrToFloatOnline(const char *p, size_t beg, size_t &end, char token) {
    if (beg == end) { return 0; }
    double r = 0.0;
    // Assume already 0-9 chars.
    for (; beg < end && p[beg] != '.'; beg++) {
        r = (r * 10.0) + (p[beg] - '0');
    }
    assert(beg == end || p[beg] == '.');
    if (beg == end) { return 0; }
    beg++;
    double frac = 0.0;

    auto prev_beg = beg;
    for (; beg < end && p[beg] != token; beg++) {
        frac = (frac * 10.0) + (p[beg] - '0');
    }
    auto frac_size = beg - prev_beg;
    static thread_local double latter_digits[] = {
            1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001
    };
    r += frac * latter_digits[frac_size];
    end = beg;
    return r;
}

#define Y_MUL (10000)
#define M_MUL (100)

// Assume YYYY-MM-DD
inline uint32_t ConvertDateToUint32(const char *date) {
    char buf[11];
    memcpy(buf, date, sizeof(char) * 11);
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

#define Y_BASE (1970)
#define M_BASE (01)
#define D_BASE (01)
#define MAX_DAYS_M (31)
#define MAX_DAYS_Y (MAX_DAYS_M * 12)

inline uint32_t ConvertDateToBucketID(const char *date) {
    char buf[11];
    memcpy(buf, date, sizeof(char) * 11);
    return MAX_DAYS_Y * (StrToInt(buf, 0, 4) - Y_BASE) +
           MAX_DAYS_M * (StrToInt(buf, 5, 7) - M_BASE)
           + (StrToInt(buf, 8, 10) - D_BASE);
}

// Assume Large Enough for "YYYY-MM-DD" (10 chars)
inline void ConvertBucketIDToDate(char *date, uint32_t val) {
    stringstream ss;
    ss << std::setw(4) << std::setfill('0') << (val / MAX_DAYS_Y + Y_BASE) << "-";
    val %= MAX_DAYS_Y;
    ss << std::setw(2) << (val / MAX_DAYS_M + M_BASE) << "-";
    val %= MAX_DAYS_M;
    ss << std::setw(2) << (val + D_BASE);
    memcpy(date, ss.str().c_str(), 10);
}

