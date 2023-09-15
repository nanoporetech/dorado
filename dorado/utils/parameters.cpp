#include "parameters.h"

#include <algorithm>
#include <thread>

namespace dorado::utils {

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_remora_threads,
                                             int max_threads) {
    ThreadAllocations allocs;
    max_threads = max_threads == 0 ? std::thread::hardware_concurrency() : max_threads;
    allocs.writer_threads = num_devices * 2;
    allocs.read_converter_threads = num_devices * 2;
    allocs.read_filter_threads = num_devices * 2;
    allocs.remora_threads = num_remora_threads;
    allocs.scaler_node_threads = num_devices * 4;
    allocs.loader_threads = num_devices;
    int total_threads_used =
            (allocs.writer_threads + allocs.read_converter_threads + allocs.read_filter_threads +
             allocs.remora_threads + allocs.scaler_node_threads + allocs.loader_threads);
    allocs.remaining_threads = max_threads - total_threads_used;
    allocs.remaining_threads = std::max(num_devices * 10, allocs.remaining_threads);
    return allocs;
};

}  // namespace dorado::utils
