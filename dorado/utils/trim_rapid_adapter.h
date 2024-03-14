#pragma once
#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

namespace dorado::utils::rapid {

struct Settings {
    // Deactivates rapid adapter trimming if false
    bool active{true};
    // The number of samples to search
    int64_t signal_len{5000};
    // The step size over signal_len
    int64_t signal_step{4};
    // The minimum length of a signal to consider searching
    int64_t signal_min_len{1500};

    // The threshold under which a rapid adapter signal will be considered
    int16_t threshold{675};
    // The threshold under which rapid adapter sample must contain at least one of
    int16_t min_threshold{500};

    // The minimum span for a rapid adapter in original samples (before stepping)
    int64_t min_span{28};
    // The minimum start point for a rapid adapter in original samples (before stepping)
    int64_t min_start{40};
    // The weight given to a candidate region at the start of the signal versus the end.
    float time_weight{100.0f};
};

Settings get_settings();

// Find the index of the end of the rapid adapter
int64_t find_rapid_adapter_trim_pos(const at::Tensor& signal, const Settings& settings);

}  // namespace dorado::utils::rapid
