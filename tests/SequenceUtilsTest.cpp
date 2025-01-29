#include "utils/sequence_utils.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <random>

#define TEST_GROUP "[seq_utils]"

using std::make_tuple;
using namespace dorado::utils;

CATCH_TEST_CASE(TEST_GROUP ": Test compute_overlap", TEST_GROUP) {
    CATCH_SECTION("Test overlaps of long identical strings") {
        auto query =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        dorado::MmTbufPtr working_buffer;
        auto overlap = compute_overlap(query, "query", target, "target", working_buffer);

        CATCH_CHECK(overlap->query_start == 0);
        CATCH_CHECK(overlap->query_end == static_cast<int>(query.size()) - 1);
        CATCH_CHECK(overlap->target_start == 0);
        CATCH_CHECK(overlap->target_end == static_cast<int>(target.size()) - 1);
    }

    CATCH_SECTION("Test overlaps of strings where one is a prefix of the other") {
        auto query = std::string(
                "TTTTTACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        auto target =
                std::string("ACGACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTT");
        dorado::MmTbufPtr working_buffer;
        auto overlap = compute_overlap(query, "query", target, "target", working_buffer);

        CATCH_CHECK(overlap->query_start == 0);
        CATCH_CHECK(overlap->query_end == static_cast<int>(target.size()) - 1);
        CATCH_CHECK(overlap->target_start == 5);
        CATCH_CHECK(overlap->target_end == static_cast<int>(query.size()) - 1);
    }
}

CATCH_TEST_CASE(TEST_GROUP ": Test base_to_int", TEST_GROUP) {
    CATCH_CHECK(base_to_int('A') == 0);
    CATCH_CHECK(base_to_int('C') == 1);
    CATCH_CHECK(base_to_int('G') == 2);
    CATCH_CHECK(base_to_int('T') == 3);
}

CATCH_TEST_CASE(TEST_GROUP ": Test sequence_to_ints", TEST_GROUP) {
    CATCH_SECTION("Test empty string") {
        auto actual_results = sequence_to_ints("");
        std::vector<int> expected_results = {};
        CATCH_CHECK(expected_results == actual_results);
    }

    CATCH_SECTION("Test single char") {
        auto actual_results = sequence_to_ints("G");
        std::vector<int> expected_results = {2};
        CATCH_CHECK(expected_results == actual_results);
    }
}

CATCH_TEST_CASE(TEST_GROUP "reverse_complement", TEST_GROUP) {
    CATCH_CHECK(dorado::utils::reverse_complement("") == "");
    CATCH_CHECK(dorado::utils::reverse_complement("ACGT") == "ACGT");
    std::srand(42);
    const std::string bases("ACGT");
    for (int i = 0; i < 10; ++i) {
        const int len = std::rand() % 20000;
        std::string temp(len, ' ');
        std::string rev_comp(len, ' ');
        for (int j = 0; j < len; ++j) {
            const int base_index = std::rand() % 4;
            char temp_base = bases.at(base_index);
            char rev_comp_base = bases.at(3 - base_index);
            // Randomly switch to lower case.
            if (rand() & 1) {
                temp_base = static_cast<char>(std::tolower(temp_base));
                rev_comp_base = static_cast<char>(std::tolower(rev_comp_base));
            }
            temp.at(j) = temp_base;
            rev_comp.at(len - 1 - j) = rev_comp_base;
        }
        CATCH_CHECK(dorado::utils::reverse_complement(temp) == rev_comp);
    }
}

CATCH_TEST_CASE(TEST_GROUP "mean_q_score", TEST_GROUP) {
    CATCH_CHECK(dorado::utils::mean_qscore_from_qstring("") == 0.0f);

    // Repeated values within range.
    std::srand(42);
    for (int q = 1; q <= 50; ++q) {
        std::string q_string(rand() % 100 + 1, '!' + static_cast<char>(q));
        CATCH_CHECK(dorado::utils::mean_qscore_from_qstring(q_string) ==
                    Catch::Approx(static_cast<float>(q)));
    }

    // Values outside normal range that will be clamped.
    CATCH_CHECK(dorado::utils::mean_qscore_from_qstring("!") == 1.0f);
    CATCH_CHECK(dorado::utils::mean_qscore_from_qstring("Z") == 50.0f);

    // Sample inputs/golden output values.
    const std::vector<std::tuple<std::string, float>> kExamples = {
            {"$$$$$%$###%&$%$$$#$$%&//*.,+((())*((&&'&$$%/.)((-3:>1(-(4NB;?C@>78?B@3", 6.27468f},
            {"464887/55.519;@=>?0..,-./*)+$&&/00)*++-//-20?@===@D:9/=<:<E@AB;98(&$%&+*", 11.61238f},
            {"33B<87ESEA41GDDSGHDC?=>:84:<?568@", 23.70278f},
            {"%$$')*(,*+78665;3378H@=>A42004.", 10.62169f}};
    for (const auto & [str, score] : kExamples) {
        CATCH_CHECK(dorado::utils::mean_qscore_from_qstring(str) == Catch::Approx(score));
    }
}

CATCH_TEST_CASE(TEST_GROUP "mean_q_score from non-zero start position", TEST_GROUP) {
    auto [str, start_pos, score] = GENERATE(table<std::string, int, float>(
            {make_tuple("####%%%%", 0, 2.88587f), make_tuple("####%%%%", 4, 4.0f)}));
    CATCH_CHECK(dorado::utils::mean_qscore_from_qstring(str.substr(start_pos)) ==
                Catch::Approx(score));
}

CATCH_TEST_CASE(TEST_GROUP ": count leading chars", TEST_GROUP) {
    auto [seq, expected] = GENERATE(table<std::string, size_t>({
            make_tuple("", 0),
            make_tuple("A", 1),
            make_tuple("C", 0),
            make_tuple("AAA", 3),
            make_tuple("AAAACGT", 4),
            make_tuple("CAGT", 0),
            make_tuple("CGTCGT", 0),
            make_tuple("CGTAAA", 0),
    }));
    CATCH_CAPTURE(seq);
    auto actual = dorado::utils::count_leading_chars(seq, 'A');
    CATCH_CHECK(actual == expected);
}

CATCH_TEST_CASE(TEST_GROUP ": count trailing chars", TEST_GROUP) {
    auto [seq, expected] = GENERATE(table<std::string, size_t>({
            make_tuple("", 0),
            make_tuple("A", 1),
            make_tuple("C", 0),
            make_tuple("AAA", 3),
            make_tuple("AAAACGT", 0),
            make_tuple("CAGT", 0),
            make_tuple("CGTCGT", 0),
            make_tuple("CGTAAA", 3),
    }));
    CATCH_CAPTURE(seq);
    auto actual = dorado::utils::count_trailing_chars(seq, 'A');
    CATCH_CHECK(actual == expected);
}

CATCH_TEST_CASE(TEST_GROUP "find rna polya", TEST_GROUP) {
    // Polya                                               |here|
    // Index                 0    5   10   15   20   25   30
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCCAAAAATCAATCA";
    const size_t expected_index = 30;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find first rna polya", TEST_GROUP) {
    // Expect to only trim the "first" polya match
    // Polya                                              |2nd |    |1st |
    // Index                 0    5   10   15   20   25   30   35   40
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCCAAAAATTTTTAAAAAC";
    const size_t expected_index = 40;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find no rna polya", TEST_GROUP) {
    // With no polyA expect to trim no bases
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCC";
    const size_t expected_index = seq.length();
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find rna polya - at start", TEST_GROUP) {
    // Polya                 |here|
    const std::string seq = "AAAAACCCCCTTTTTCCCCCTTTTTCCCCC";
    const size_t expected_index = 0;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find rna polya - straddle search area", TEST_GROUP) {
    // PolyA continues over the maximum of 200 bases
    const std::string seq(210, 'A');
    const size_t expected_index = 10;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find rna polya - outside search", TEST_GROUP) {
    // PolyA beyond the search range
    const std::string seq = std::string(5, 'A') + std::string(200, 'T');
    const size_t expected_index = seq.length();
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE(TEST_GROUP "find rna polya - within search", TEST_GROUP) {
    using S = std::string;
    const S seq = S(100, 'A') + S(155, 'C') + S(10, 'A') + S(100, 'C');
    const size_t expected_index = 255;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CATCH_CHECK(expected_index == res);
}

CATCH_TEST_CASE("Test sequence to move table index", TEST_GROUP) {
    CATCH_SECTION("Happy path") {
        // ----------------   seq index:   0, 1,  ,  ,  , 2, 3, 4,  ,  , 5,  , 6, 7,
        // ---------------- moves index:   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16
        const std::vector<uint8_t> move = {1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0};
        const size_t seq_size = move_cum_sums(move).back();

        auto [seq_index, expected] = GENERATE(table<size_t, int64_t>({
                std::make_tuple(0, 0),
                std::make_tuple(1, 1),
                std::make_tuple(2, 5),
                std::make_tuple(3, 6),
                std::make_tuple(4, 7),
                std::make_tuple(5, 10),
                std::make_tuple(6, 12),
                std::make_tuple(7, 13),
        }));

        CATCH_CAPTURE(seq_index);
        const auto res = sequence_to_move_table_index(move, seq_index, seq_size);
        CATCH_CHECK(expected == res);
    }

    CATCH_SECTION("Empty moves") {
        const std::vector<uint8_t> move = {};
        const auto res = sequence_to_move_table_index(move, 0, 0);
        CATCH_CHECK(res < 0);
    }

    CATCH_SECTION("Bad sequence index") {
        const std::vector<uint8_t> move = {0, 1, 0, 1, 0};
        const size_t seq_size = move_cum_sums(move).back();
        const auto res = sequence_to_move_table_index(move, seq_size + 1, seq_size);
        CATCH_CHECK(res < 0);
    }

    CATCH_SECTION("Bad sequence size") {
        const std::vector<uint8_t> move = {0, 1, 0, 1, 0};
        const size_t bad_seq_size = move.size() + 1;
        const auto res = sequence_to_move_table_index(move, 0, bad_seq_size);
        CATCH_CHECK(res < 0);
    }
}

CATCH_TEST_CASE(TEST_GROUP "reverse_seq_to_sig_map", TEST_GROUP) {
    auto reverse_2_passes = [](std::vector<uint64_t> & map, size_t signal_length) {
        std::reverse(map.begin(), map.end());
        std::transform(map.begin(), map.end(), map.begin(),
                       [signal_length](uint64_t v) { return signal_length - v; });
    };

    // Generate a fake map.
    std::minstd_rand rng(Catch::rngSeed());
    std::uniform_int_distribution<uint64_t> dist;
    using Range = decltype(dist)::param_type;
    auto generate_map = [&rng, &dist](uint64_t signal_length) {
        std::vector<uint64_t> map;
        uint64_t current_idx = 0;
        while (current_idx < signal_length) {
            const auto range = Range(1, signal_length - current_idx);
            const auto step = dist(rng, range);
            current_idx += step;
            map.push_back(current_idx);
        }
        if (signal_length != 0) {
            CATCH_REQUIRE(!map.empty());
            CATCH_REQUIRE(map.back() == signal_length);
        }
        return map;
    };

    // Check that both approaches match.
    for (size_t length = 0; length < 100; length++) {
        CATCH_CAPTURE(length);
        // Both act in-place.
        auto seq_to_sig_map = generate_map(length);
        auto reversed_map = seq_to_sig_map;
        reverse_2_passes(reversed_map, length);
        reverse_seq_to_sig_map(seq_to_sig_map, length);
        CATCH_CHECK(seq_to_sig_map == reversed_map);
    }

    // Benchmark it.
#if DORADO_ENABLE_BENCHMARK_TESTS
    const auto length = GENERATE(100, 10'000, 1'000'000);
    std::vector<uint64_t> fake_map(length);
    std::iota(fake_map.begin(), fake_map.end(), 0);
    CATCH_BENCHMARK(fmt::format("reverse_2_passes, {} bases", length)) {
        reverse_2_passes(fake_map, length);
    };
    CATCH_BENCHMARK(fmt::format("reverse_seq_to_sig_map, {} bases", length)) {
        reverse_seq_to_sig_map(fake_map, length);
    };
#endif  // DORADO_ENABLE_BENCHMARK_TESTS
}
