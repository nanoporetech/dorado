#pragma once

#include <string>

namespace dorado::utils {

struct DefaultParameters {
    int batchsize{0};
    int chunksize{10000};
    int overlap{500};
    int num_runners{2};

    // Minimum length for a sequence to be outputted.
    size_t min_sequence_length{5};
};

static const DefaultParameters default_parameters{};

struct ThreadAllocations {
    int writer_threads{0};
    int read_converter_threads{0};
    int read_filter_threads{0};
    int modbase_threads{0};
    int scaler_node_threads{0};
    int splitter_node_threads{0};
    int loader_threads{0};
    int aligner_threads{0};
    int barcoder_threads{0};
    int adapter_threads{0};
};

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_modbase_threads,
                                             bool enable_aligner,
                                             bool enable_barcoder,
                                             bool adapter_trimming);

}  // namespace dorado::utils
