#include "parameters.h"

#include <algorithm>
#include <thread>

namespace dorado::utils {

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_remora_threads,
                                             bool enable_aligner,
                                             bool enable_barcoder,
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
    int remaining_threads = max_threads - total_threads_used;
    remaining_threads = std::max(num_devices * 10, remaining_threads);
    // Divide up work equally between the aligner and barcoder nodes if both are enabled,
    // otherwise both get all the remaining threads.
    if (enable_aligner || enable_barcoder) {
        allocs.aligner_threads =
                remaining_threads * enable_aligner / (enable_aligner + enable_barcoder);
        allocs.barcoder_threads =
                remaining_threads * enable_barcoder / (enable_aligner + enable_barcoder);
    }
    return allocs;
};

}  // namespace dorado::utils
