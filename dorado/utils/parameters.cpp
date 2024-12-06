#include "parameters.h"

#include <algorithm>
#include <thread>

namespace dorado::utils {

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_modbase_threads,
                                             bool enable_aligner,
                                             bool enable_barcoder,
                                             bool adapter_trimming) {
    const int max_threads = std::thread::hardware_concurrency();
    ThreadAllocations allocs;
    allocs.writer_threads = num_devices * 2;
    allocs.read_converter_threads = num_devices * 2;
    allocs.read_filter_threads = num_devices * 2;
    allocs.modbase_threads = num_devices * num_modbase_threads;
    allocs.scaler_node_threads = num_devices * 4;
    allocs.splitter_node_threads = num_devices;
    allocs.loader_threads = num_devices;
    int total_threads_used =
            (allocs.writer_threads + allocs.read_converter_threads + allocs.read_filter_threads +
             allocs.modbase_threads + allocs.scaler_node_threads + allocs.loader_threads +
             allocs.splitter_node_threads);
    int remaining_threads = max_threads - total_threads_used;
    remaining_threads = std::max(num_devices * 10, remaining_threads);
    // Divide up work equally between the aligner, barcoder, and adapter-trimming nodes, or whatever
    // subset of them are enabled.
    if (enable_aligner || enable_barcoder || adapter_trimming) {
        int number_enabled = int(enable_aligner) + int(enable_barcoder) + int(adapter_trimming);
        allocs.aligner_threads = remaining_threads * int(enable_aligner) / number_enabled;
        allocs.barcoder_threads = remaining_threads * int(enable_barcoder) / number_enabled;
        allocs.adapter_threads = remaining_threads * int(adapter_trimming) / number_enabled;
    }
    return allocs;
};

}  // namespace dorado::utils
