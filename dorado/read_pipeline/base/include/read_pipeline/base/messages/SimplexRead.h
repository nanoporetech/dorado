#pragma once

#include "ReadCommon.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace dorado {

// Class representing a simplex read, including raw data
class SimplexRead {
public:
    ReadCommon read_common;

    float digitisation;  // Loaded from source file
    float range;         // Loaded from source file
    float offset;        // Loaded from source file

    uint64_t get_end_time_ms() const;

    float scaling;  // Scale factor applied to convert raw integers from sequencer into pore current values

    uint64_t start_sample;
    uint64_t end_sample;
    uint64_t run_acquisition_start_time_ms;
    // Calculate mean Q-score from this position onwards if read is
    // a short read.

    std::atomic_size_t num_duplex_candidate_pairs{0};

    // This is atomic because multiple threads can write to it at the same time.
    // For example, if a read (call it 2) is in the cache, and is selected as a potential pair match by two incoming reads (1 and 3) on two other threads, these threads can both update `is_duplex_parent` at the same time.
    std::atomic_bool is_duplex_parent{false};

    // Track the previous/next read fom the same channel/mux.
    std::string prev_read;
    std::string next_read;

    // Loaded from V4+ pod5 files
    float open_pore_level{std::numeric_limits<float>::quiet_NaN()};
    // Loaded from V5+ pod5 files
    float expected_open_pore_level{std::numeric_limits<float>::quiet_NaN()};
    float selected_read_level{std::numeric_limits<float>::quiet_NaN()};
};

using SimplexReadPtr = std::unique_ptr<SimplexRead>;

}  // namespace dorado
