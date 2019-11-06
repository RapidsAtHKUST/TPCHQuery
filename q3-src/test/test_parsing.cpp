//
// Created by yche on 11/6/19.
//

#include "../parsing_util.h"
#include "../util/log.h"

int main() {
//    const char *my_chars = "4294967295";
    const char *my_chars = "3000000000";
    size_t end = strlen(my_chars);
    auto integer = StrToIntOnline(my_chars, 0, end, '\0');
    log_info("%u", integer);
}
