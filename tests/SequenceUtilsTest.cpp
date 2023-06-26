#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#include <cstdlib>

#define TEST_GROUP "[utils]"

using std::make_tuple;
using namespace dorado::utils;

TEST_CASE(TEST_GROUP ": Test base_to_int") {
    CHECK(base_to_int('A') == 0);
    CHECK(base_to_int('C') == 1);
    CHECK(base_to_int('G') == 2);
    CHECK(base_to_int('T') == 3);
}

TEST_CASE(TEST_GROUP ": Test sequence_to_ints") {
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

TEST_CASE(TEST_GROUP "reverse_complement") {
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
            temp.at(j) = bases.at(base_index);
            rev_comp.at(len - 1 - j) = bases.at(3 - base_index);
        }
        CHECK(dorado::utils::reverse_complement(temp) == rev_comp);
    }
}

TEST_CASE(TEST_GROUP "mean_q_score") {
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

TEST_CASE(TEST_GROUP "mean_q_score from non-zero start position") {
    CHECK(dorado::utils::mean_qscore_from_qstring("") == 0.0f);

    auto [str, start_pos, score] = GENERATE(table<std::string, int, float>(
            {make_tuple("####%%%%", 0, 2.88587f), make_tuple("####%%%%", 4, 7.0103f)}));
    CHECK(dorado::utils::mean_qscore_from_qstring(str, start_pos) == Approx(score));

    CHECK_THROWS_AS(dorado::utils::mean_qscore_from_qstring("####", 10), std::runtime_error);
}
