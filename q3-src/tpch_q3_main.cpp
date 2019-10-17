//
// Created by yche on 10/10/19.
//

#include <chrono>
#include <thread>
#include <cassert>

#include <omp.h>

#include "util/log.h"
#include "util/program_options/popl.h"
#include "util/primitives/parasort_cmp.h"
#include "util/timer.h"
#include "util/util.h"
#include "util/pretty_print.h"
#include "util/primitives/primitives.h"
#include "util/primitives/libpopcnt.h"
#include "util/primitives/boolarray.h"
#include "util/primitives/blockingconcurrentqueue.h"
#include "util/primitives/barrier.h"

using namespace std;
using namespace popl;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    OptionParser op("Allowed options");
    auto string_option = op.add<Value<std::string>>("f", "file-path", "the graph bin file path");
    op.parse(argc, argv);

    using Edge = pair<int32_t, int32_t>;
    Timer global_timer;
    if (string_option->is_set()) {
        log_info("OK");
        sleep(1);
        log_info("OK");
    }
    thread t([]() {
        log_info("Test");
    });
    t.join();
}