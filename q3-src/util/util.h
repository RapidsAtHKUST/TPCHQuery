#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstring>
#include <cstdlib>
#include <cstdint>

#include <string>
#include <iomanip>
#include <locale>
#include <sstream>

#include "log.h"

using namespace std;

#define Y_MUL (10000)
#define M_MUL (100)

// Assume YYYY-MM-DD
uint32_t ConvertDateToUint32(const char *date) {
    char buf[11];
    memcpy(buf, date, sizeof(char) * 11);
    buf[4] = '\0';
    buf[7] = '\0';
    buf[10] = '\0';
    return Y_MUL * atoi(buf) + M_MUL * atoi(buf + 5) + atoi(buf + 8);
}

// Asssume Large Enough for "YYYY-MM-DD" (10 chars)
void ConvertUint32ToDate(char *date, uint32_t val) {
    char buf[10];
    stringstream ss;
    ss << std::setw(4) << std::setfill('0') << val / Y_MUL << "-";
    val %= Y_MUL;
    ss << std::setw(2) << val / M_MUL << "-";
    val %= M_MUL;
    ss << std::setw(2) << val;
    memcpy(date, ss.str().c_str(), 10);
}


template<class T>
std::string FormatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

inline void reset(std::stringstream &stream) {
    const static std::stringstream initial;

    stream.str(std::string());
    stream.clear();
    stream.copyfmt(initial);
}

inline size_t file_size(const char *file_name) {
    struct stat st;
    stat(file_name, &st);
    size_t size = st.st_size;
    return size;
}

inline int parseLine(char *line) {
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char *p = line;
    while (*p < '0' || *p > '9') p++;
    line[i - 3] = '\0';
    i = atoi(p);
    return i;
}

inline int getValue() { //Note: this value is in KB!
    FILE *file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}