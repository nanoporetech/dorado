#pragma once

#include <algorithm>
#include <string>
#include <thread>

namespace dorado::utils {

struct DefaultParameters {
#if !DORADO_GPU_BUILD
    std::string device{"cpu"};
#elif defined(__APPLE__)
    std::string device{"metal"};
#else
    std::string device{"cuda:all"};
#endif
    int batchsize{0};
    int chunksize{10000};
    int overlap{500};
    int num_runners{2};
#ifdef DORADO_TX2
    int remora_batchsize{128};
#else
    int remora_batchsize{1024};
#endif
    int remora_threads{4};
    float methylation_threshold{5.0f};
};

static const DefaultParameters default_parameters{};

struct ThreadAllocations {
    ThreadAllocations(int num_devices, int num_remora_threads, int max_threads = 0) {
        max_threads = max_threads == 0 ? std::thread::hardware_concurrency() : max_threads;
        writer_threads = num_devices * 2;
        read_converter_threads = num_devices * 2;
        read_filter_threads = num_devices * 2;
        remora_threads = num_remora_threads;
        scaler_node_threads = num_devices * 4;
        loader_threads = num_devices;
        int total_threads_used = (writer_threads + read_converter_threads + read_filter_threads +
                                  remora_threads + scaler_node_threads + loader_threads);
        int remaining_threads = max_threads - total_threads_used;
        aligner_threads = std::max(num_devices * 10, remaining_threads);
    }

    int writer_threads{0};
    int read_converter_threads{0};
    int read_filter_threads{0};
    int remora_threads{0};
    int scaler_node_threads{0};
    int loader_threads{0};
    int aligner_threads{0};
};

inline ThreadAllocations default_thread_allocations(int num_devices,
                                                    int num_remora_threads,
                                                    int max_threads = 0) {
    return ThreadAllocations(num_devices, num_remora_threads, max_threads);
}

}  // namespace dorado::utils
