#include "trim_rapid_adapter.h"

#include "utils/dev_utils.h"

#include <ATen/ATen.h>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <toml/get.hpp>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace dorado::utils::rapid {

// Checks that Settings has valid values
bool validate_settings(const Settings& s) {
    bool is_valid = true;

    for (const auto v : {s.signal_len, s.signal_min_len, s.signal_step, s.min_span, s.min_start}) {
        if (v < (int64_t)1) {
            spdlog::error("trim_rapid_adapter - signal values must be positive found: ", v);
            is_valid = false;
        }
    }

    for (const auto v : {s.threshold, s.min_threshold}) {
        if (v < (int16_t)1) {
            spdlog::error("trim_rapid_adapter - threshold values must be positive found: ", v);
            is_valid = false;
        }
    }

    if (s.signal_len <= s.signal_min_len) {
        spdlog::error("trim_rapid_adapter - signal_len({}) <= signal_min_len({})", s.signal_len,
                      s.signal_min_len);
        is_valid = false;
    }

    if (s.signal_len < s.signal_step) {
        spdlog::error("trim_rapid_adapter - signal_len({}) < signal_step({})", s.signal_len,
                      s.signal_step);
        is_valid = false;
    }

    if (s.signal_len <= s.min_span) {
        spdlog::error("trim_rapid_adapter - signal_len({}) <= min_span({})", s.signal_len,
                      s.min_span);
        is_valid = false;
    }

    if (s.signal_len <= s.min_start) {
        spdlog::error("trim_rapid_adapter - signal_len({}) <= min_start({})", s.signal_len,
                      s.min_start);
        is_valid = false;
    }

    if (s.threshold < 0 || s.min_threshold < 0) {
        spdlog::error("trim_rapid_adapter - threshold({}) < 0 or min_threshold({}) < 0",
                      s.threshold, s.min_threshold);
    }

    if (s.time_weight < 1.0f) {
        spdlog::error("trim_rapid_adapter - time_weight({}) < 1.0", s.time_weight);
        is_valid = false;
    }

    return is_valid;
}

Settings load_rapid_trim_config(const std::filesystem::path& path) {
    Settings s;

    spdlog::trace("load_rapid_trim_config path: {}", path.string());
    const toml::value cfg = toml::parse(path);

    if (!cfg.contains("rapid")) {
        throw std::runtime_error("load_rapid_trim_config toml file missing expected `rapid` table");
    }

    const auto& c = toml::find(cfg, "rapid");
    s.active = toml::find_or<int>(c, "active", s.active) > 0;

    if (!s.active) {
        return s;
    }

    s.signal_len = toml::find_or<int64_t>(c, "signal_len", s.signal_len);
    s.signal_min_len = toml::find_or<int64_t>(c, "signal_min_len", s.signal_min_len);
    s.signal_step = toml::find_or<int64_t>(c, "signal_step", s.signal_step);
    s.threshold = static_cast<int16_t>(toml::find_or<int>(c, "threshold", s.threshold));
    s.min_threshold = static_cast<int16_t>(toml::find_or<int>(c, "min_threshold", s.min_threshold));
    s.min_span = toml::find_or<int64_t>(c, "min_span", s.min_span);
    s.min_start = toml::find_or<int64_t>(c, "min_start", s.min_start);
    s.time_weight = static_cast<float>(
            toml::find_or<double>(c, "time_weight", static_cast<unsigned long>(s.time_weight)));

    return s;
}

// Get settings for rapid adapter trimming
Settings get_settings() {
    Settings s;
    bool use_toml = utils::get_dev_opt<int>("rapid_toml", 0) > 0;
    if (use_toml) {
        const auto path = std::filesystem::current_path() / "rapid.toml";
        if (!std::filesystem::exists(path)) {
            spdlog::error("`rapid.toml` file does not exist at: {}", path.string());
            throw std::runtime_error("`rapid.toml` file does not exist");
        }

        s = load_rapid_trim_config(path);
        if (!validate_settings(s)) {
            throw std::logic_error("invalid rapid adapter trim settings");
        }
    }

    if (!s.active) {
        spdlog::debug("rapid adapter trimming deactivated");
    }

    return s;
};

int64_t find_rapid_adapter_trim_pos(const at::Tensor& signal, const Settings& s) {
    if (!s.active) {
        return -1;
    }

    const int64_t signal_size = signal.size(0);
    if (signal_size < s.signal_min_len) {
        spdlog::trace("trim_rapid_adapter signal_size < signal_min_len - {} < {} - skip",
                      signal_size, s.signal_min_len);
        return -1;
    }

    bool is_region_active = false;
    bool is_min_below_threshold = false;
    uint64_t vol = 0;
    uint64_t best_vol = 0;
    int64_t start = 0;
    int64_t best_start = 0;
    int64_t best_end = 0;

    auto signal_a = signal.accessor<int16_t, 1>();

    // Compute the division once here
    const float time_weight_coeff =
            static_cast<float>(s.time_weight) / static_cast<float>(signal_size);

    for (int64_t i = s.min_start; i < signal_size; i += s.signal_step) {
        const auto sample = signal_a[i];

        // Compute the volume of a contiguous region under the threshold
        if (sample < s.threshold) {
            // Start a new region
            if (!is_region_active) {
                start = i;
                is_region_active = true;
            }

            // Check at least one sample is below the stricter threshold
            if (sample < s.min_threshold) {
                is_min_below_threshold = true;
            }

            // (threshold - sample) always +ve as sample < threshold
            // This value should not overflow:
            // max_vol := signal_len * threshold^2 * time_weight
            // max_vol := 1e4 * 1e6 * 1e3 ~> 1e13
            const auto delta = s.threshold - sample;
            vol += delta * delta;
        } else {
            // Check span and min threshold
            if (((i - start) >= s.min_span) && is_min_below_threshold) {
                // Compute time weighted volume to significantly up-weight regions early in the signal
                vol *= static_cast<uint64_t>(time_weight_coeff * (signal_size - i));

                if (vol > best_vol) {
                    best_vol = vol;
                    best_start = start;
                    best_end = i;
                }
            }

            // Reset values for the next region
            is_region_active = false;
            is_min_below_threshold = false;
            vol = 0;
        }
    }

    if (best_start <= s.min_start || best_end >= signal_size - 1 || best_vol == 0) {
        return -1;
    }

    return best_end;
}

}  // namespace dorado::utils::rapid