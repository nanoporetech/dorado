#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#include <cstdint>
#include <cstdlib>
#include <optional>

#define TEST_GROUP "[seq_utils]"

using std::make_tuple;
using namespace dorado::utils;

TEST_CASE(TEST_GROUP ": Test base_to_int", TEST_GROUP) {
    CHECK(base_to_int('A') == 0);
    CHECK(base_to_int('C') == 1);
    CHECK(base_to_int('G') == 2);
    CHECK(base_to_int('T') == 3);
}

TEST_CASE(TEST_GROUP ": Test sequence_to_ints", TEST_GROUP) {
    SECTION("Test empty string") {
        auto actual_results = sequence_to_ints("");
        std::vector<int> expected_results = {};
        CHECK(expected_results == actual_results);
    }

    SECTION("Test single char") {
        auto actual_results = sequence_to_ints("G");
        std::vector<int> expected_results = {2};
        CHECK(expected_results == actual_results);
    }
}

TEST_CASE(TEST_GROUP "reverse_complement", TEST_GROUP) {
    CHECK(dorado::utils::reverse_complement("") == "");
    CHECK(dorado::utils::reverse_complement("ACGT") == "ACGT");
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
        CHECK(dorado::utils::reverse_complement(temp) == rev_comp);
    }
}

TEST_CASE(TEST_GROUP "mean_q_score", TEST_GROUP) {
    CHECK(dorado::utils::mean_qscore_from_qstring("") == 0.0f);

    // Repeated values within range.
    std::srand(42);
    for (int q = 1; q <= 50; ++q) {
        std::string q_string(rand() % 100 + 1, '!' + static_cast<char>(q));
        CHECK(dorado::utils::mean_qscore_from_qstring(q_string) == Approx(static_cast<float>(q)));
    }

    // Values outside normal range that will be clamped.
    CHECK(dorado::utils::mean_qscore_from_qstring("!") == 1.0f);
    CHECK(dorado::utils::mean_qscore_from_qstring("Z") == 50.0f);

    // Sample inputs/golden output values.
    const std::vector<std::tuple<std::string, float>> kExamples = {
            {"$$$$$%$###%&$%$$$#$$%&//*.,+((())*((&&'&$$%/.)((-3:>1(-(4NB;?C@>78?B@3", 6.27468f},
            {"464887/55.519;@=>?0..,-./*)+$&&/00)*++-//-20?@===@D:9/=<:<E@AB;98(&$%&+*", 11.61238f},
            {"33B<87ESEA41GDDSGHDC?=>:84:<?568@", 23.70278f},
            {"%$$')*(,*+78665;3378H@=>A42004.", 10.62169f}};
    for (const auto& [str, score] : kExamples) {
        CHECK(dorado::utils::mean_qscore_from_qstring(str) == Approx(score));
    }
}

TEST_CASE(TEST_GROUP "mean_q_score from non-zero start position", TEST_GROUP) {
    auto [str, start_pos, score] = GENERATE(table<std::string, int, float>(
            {make_tuple("####%%%%", 0, 2.88587f), make_tuple("####%%%%", 4, 4.0f)}));
    CHECK(dorado::utils::mean_qscore_from_qstring(str.substr(start_pos)) == Approx(score));
}

TEST_CASE(TEST_GROUP ": count leading chars", TEST_GROUP) {
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
    CAPTURE(seq);
    auto actual = dorado::utils::count_leading_chars(seq, 'A');
    CHECK(actual == expected);
}

TEST_CASE(TEST_GROUP ": count trailing chars", TEST_GROUP) {
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
    CAPTURE(seq);
    auto actual = dorado::utils::count_trailing_chars(seq, 'A');
    CHECK(actual == expected);
}

TEST_CASE(TEST_GROUP "find rna polya", TEST_GROUP) {
    // Polya                                               |here|
    // Index                 0    5   10   15   20   25   30
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCCAAAAATCAATCA";
    const size_t expected_index = 30;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find first rna polya", TEST_GROUP) {
    // Expect to only trim the "first" polya match
    // Polya                                              |2nd |    |1st |
    // Index                 0    5   10   15   20   25   30   35   40
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCCAAAAATTTTTAAAAAC";
    const size_t expected_index = 40;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find no rna polya", TEST_GROUP) {
    // With no polyA expect to trim no bases
    const std::string seq = "TTTTTCCCCCTTTTTCCCCCTTTTTCCCCC";
    const size_t expected_index = seq.length();
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find rna polya - at start", TEST_GROUP) {
    // Polya                 |here|
    const std::string seq = "AAAAACCCCCTTTTTCCCCCTTTTTCCCCC";
    const size_t expected_index = 0;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find rna polya - straddle search area", TEST_GROUP) {
    // PolyA continues over the maximum of 200 bases
    const std::string seq(210, 'A');
    const size_t expected_index = 10;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find rna polya - outside search", TEST_GROUP) {
    // PolyA beyond the search range
    const std::string seq = std::string(5, 'A') + std::string(200, 'T');
    const size_t expected_index = seq.length();
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE(TEST_GROUP "find rna polya - within search", TEST_GROUP) {
    using S = std::string;
    const S seq = S(100, 'A') + S(155, 'C') + S(10, 'A') + S(100, 'C');
    const size_t expected_index = 255;
    const size_t res = dorado::utils::find_rna_polya(seq);
    CHECK(expected_index == res);
}

TEST_CASE("Test sequence to move table index", TEST_GROUP) {
    SECTION("Happy path") {
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

        CAPTURE(seq_index);
        const auto res = sequence_to_move_table_index(move, seq_index, seq_size);
        CHECK(expected == res);
    }

    SECTION("Empty moves") {
        const std::vector<uint8_t> move = {};
        const auto res = sequence_to_move_table_index(move, 0, 0);
        CHECK(res < 0);
    }

    SECTION("Bad sequence index") {
        const std::vector<uint8_t> move = {0, 1, 0, 1, 0};
        const size_t seq_size = move_cum_sums(move).back();
        const auto res = sequence_to_move_table_index(move, seq_size + 1, seq_size);
        CHECK(res < 0);
    }

    SECTION("Bad sequence size") {
        const std::vector<uint8_t> move = {0, 1, 0, 1, 0};
        const size_t bad_seq_size = move.size() + 1;
        const auto res = sequence_to_move_table_index(move, 0, bad_seq_size);
        CHECK(res < 0);
    }
}
