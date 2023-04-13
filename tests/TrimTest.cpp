#include "utils/trim.h"

#include <catch2/catch.hpp>

#include <random>

using Slice = torch::indexing::Slice;

#define TEST_GROUP "[utils][trim]"

TEST_CASE("Test trim signal", TEST_GROUP) {
    constexpr int signal_len = 2000;

    std::mt19937 gen{42};
    std::normal_distribution<float> rng{0, 1};

    std::vector<float> signal(signal_len);
    std::generate(signal.begin(), signal.end(), [&]() { return rng(gen); });

    // add a peak just after the start
    for (int i = 1; i < 55; ++i) {
        signal[i] += 5;
    }

    auto signal_tensor = torch::from_blob(const_cast<float*>(signal.data()), {signal_len});

    SECTION("Default trim") {
        int pos = dorado::utils::trim(signal_tensor);

        // pos 55 is in the second window of 40 samples, after a min_trim of 10
        int expected_pos = 90;
        CHECK(pos == expected_pos);

        // begin with a plateau instead of a peak, should still find the same end
        signal[0] += 5;
        CHECK(pos == expected_pos);
    }

    SECTION("Reduced window size") {
        int pos = dorado::utils::trim(signal_tensor, 2.4, 10);

        int expected_pos = 60;
        CHECK(pos == expected_pos);
    }

    SECTION("All signal below threshold") {
        int pos = dorado::utils::trim(signal_tensor, 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }

    SECTION("All signal above threshold") {
        std::fill(std::begin(signal), std::end(signal), 100.f);
        int pos = dorado::utils::trim(signal_tensor, 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }

    SECTION("Peak beyond max samples") {
        for (int i = 500; i < 555; ++i) {
            signal[i] += 50;
        }

        int pos = dorado::utils::trim(signal_tensor.index({Slice(torch::indexing::None, 400)}), 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }
}
