#include "utils/trim.h"

#include <catch2/catch.hpp>

#include <random>

using Slice = torch::indexing::Slice;
using namespace dorado;

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

    auto signal_tensor = torch::from_blob(const_cast<float *>(signal.data()), {signal_len});

    SECTION("Default trim") {
        int pos = utils::trim(signal_tensor);

        // pos 55 is in the second window of 40 samples, after a min_trim of 10
        int expected_pos = 90;
        CHECK(pos == expected_pos);

        // begin with a plateau instead of a peak, should still find the same end
        signal[0] += 5;
        CHECK(pos == expected_pos);
    }

    SECTION("Reduced window size") {
        int pos = utils::trim(signal_tensor, 2.4, 10);

        int expected_pos = 60;
        CHECK(pos == expected_pos);
    }

    SECTION("All signal below threshold") {
        int pos = utils::trim(signal_tensor, 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }

    SECTION("All signal above threshold") {
        std::fill(std::begin(signal), std::end(signal), 100.f);
        int pos = utils::trim(signal_tensor, 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }

    SECTION("Peak beyond max samples") {
        for (int i = 500; i < 555; ++i) {
            signal[i] += 50;
        }

        int pos = utils::trim(signal_tensor.index({Slice(torch::indexing::None, 400)}), 24);

        int expected_pos = 10;  // minimum trim value
        CHECK(pos == expected_pos);
    }
}

TEST_CASE("Test trim sequence", TEST_GROUP) {
    const std::string seq = "TEST_SEQ";

    SECTION("Test empty sequence") { CHECK(utils::trim_sequence("", {0, 100}) == ""); }

    SECTION("Trim nothing") { CHECK(utils::trim_sequence(seq, {0, seq.length()}) == seq); }

    SECTION("Trim part of the sequence") {
        CHECK(utils::trim_sequence(seq, {5, seq.length()}) == "SEQ");
    }

    SECTION("Trim whole sequence") { CHECK(utils::trim_sequence(seq, {0, 0}) == ""); }
}

TEST_CASE("Test trim quality vector", TEST_GROUP) {
    const std::vector<uint8_t> qual = {30, 30, 56, 60, 72, 10};

    SECTION("Test empty sequence") { CHECK(utils::trim_quality({}, {0, 20}).size() == 0); }

    SECTION("Trim nothing") { CHECK(utils::trim_quality(qual, {0, qual.size()}) == qual); }

    SECTION("Trim part of the sequence") {
        const std::vector<uint8_t> expected = {10};
        CHECK(utils::trim_quality(qual, {5, qual.size()}) == expected);
    }

    SECTION("Trim whole sequence") { CHECK(utils::trim_quality(qual, {0, 0}).size() == 0); }
}

TEST_CASE("Test trim move table", TEST_GROUP) {
    const std::vector<uint8_t> move = {1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1};

    SECTION("Trim nothing") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {0, move.size()});
        CHECK(ts == 0);
        CHECK(trimmed_table == move);
    }

    SECTION("Trim part of the sequence") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {3, 5});
        CHECK(ts == 6);
        const std::vector<uint8_t> expected = {1, 1, 0, 0};
        CHECK(trimmed_table == expected);
    }

    SECTION("Trim whole sequence") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {0, 0});
        CHECK(ts == 0);
        CHECK(trimmed_table.size() == 0);
    }
}

TEST_CASE("Test trim mod base info", TEST_GROUP) {
    const std::string modbase_str = "MM:Z:C+h?,4,1,6,1,0;C+m?,3,1,2,1;C+x?,1,17;";
    const std::vector<int8_t> modbase_probs = {2, 3, 4, 5, 6, 10, 11, 12, 13, 20, 21};

    SECTION("Trim nothing") {
        auto [str, probs] = utils::trim_modbase_info(modbase_str, modbase_probs, {0, 200});
        CHECK(str == modbase_str);
        CHECK(probs == modbase_probs);
    }

    SECTION("Trim part of the sequence") {
        // This position tests 3 cases together -
        // in the first mod, trimming truncates skips from 6 to 5
        // in the second mod, the trimmed seq start from a mod
        // the third mod is eliminated
        auto [str, probs] = utils::trim_modbase_info(modbase_str, modbase_probs, {8, 18});
        CHECK(str == "MM:Z:C+h?,5,1,0;C+m?,0,1;");
        const std::vector<int8_t> expected = {4, 5, 6, 12, 13};
        CHECK(probs == expected);
    }

    SECTION("Trim whole sequence") {
        auto [str, probs] = utils::trim_modbase_info(modbase_str, modbase_probs, {8, 8});
        CHECK(str == "");
        CHECK(probs.size() == 0);
    }
}
