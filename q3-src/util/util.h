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

// utility function
inline bool file_exists(const std::string name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
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