#include "utils/trim_rapid_adapter.h"

#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <catch2/catch.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dorado::utils::rapid;

#define TEST_GROUP "[utils][trim][rapid]"

namespace {

const size_t signal_len = 4000;

// Returns a synthetic signal of length 4000 with a `baseline` constant value
// and each level(length, value) being set sequentially.
std::vector<int16_t> level_signal(const int16_t baseline,
                                  const std::vector<std::pair<size_t, int16_t>> &levels) {
    std::vector<int16_t> signal(signal_len, baseline);
    size_t i = 0;
    for (const auto &pair : levels) {
        const auto [len, value] = pair;
        if (i + len >= signal_len) {
            throw std::runtime_error("Index out of bounds in `level_signal`");
        }
        std::fill_n(signal.begin() + i, len, value);
        i += len;
    }
    return signal;
}

at::Tensor to_tensor(const std::vector<int16_t> &signal) {
    if (signal.size() != signal_len) {
        throw std::runtime_error("to_tensor expected input size of :" + std::to_string(signal_len));
    }
    return at::from_blob(const_cast<int16_t *>(signal.data()), {signal_len},
                         at::TensorOptions().dtype(at::kShort));
}

}  // namespace

TEST_CASE("Test trim rapid adapter signal", TEST_GROUP) {
    const Settings s;

    const int16_t high = s.threshold + 1;
    const int16_t mid = s.threshold - 1;
    const int16_t low = s.min_threshold - 1;

    SECTION("rapid adapter") {
        const auto signal = level_signal(high, {{100, high}, {100, low}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res == 200);
    }

    SECTION("double rapid adapter - select first") {
        const auto signal = level_signal(high, {{100, high}, {100, low}, {100, high}, {100, low}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res == 200);
    }

    SECTION("no rapid adapter - missing rapid adapter") {
        const auto signal = level_signal(high, {});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res < 0);
    }

    SECTION("no rapid adapter - low start") {
        const auto signal = level_signal(high, {{100, low}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res < 0);
    }

    SECTION("no rapid adapter - no minima") {
        const auto signal = level_signal(high, {{100, mid}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res < 0);
    }

    SECTION("rapid adapter - with minima") {
        const auto signal = level_signal(high, {{400, high}, {100, mid}, {1, low}, {99, mid}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), s);
        CHECK(res == 600);
    }

    SECTION("no rapid adapter - short signal") {
        const size_t short_len = 100;
        std::vector<int16_t> vec(short_len, high);
        const auto signal = at::from_blob(const_cast<int16_t *>(vec.data()), {short_len},
                                          at::TensorOptions().dtype(at::kShort));
        const auto res = find_rapid_adapter_trim_pos(signal, s);
        CHECK(res < 0);
    }

    SECTION("no rapid adapter - signal too short") {
        Settings short_settings;
        short_settings.signal_min_len = 5000;
        const auto signal = level_signal(high, {{100, high}, {100, low}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), short_settings);
        CHECK(res < 0);
    }

    SECTION("disabled via settings") {
        Settings inactive_settings;
        inactive_settings.active = false;
        const auto signal = level_signal(high, {{100, high}, {100, low}});
        const auto res = find_rapid_adapter_trim_pos(to_tensor(signal), inactive_settings);
        CHECK(res < 0);
    }
}