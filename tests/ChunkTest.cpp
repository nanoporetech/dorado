#include "read_pipeline/base/chunk.h"

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <span>

#define TEST_GROUP "[utils]"

CATCH_TEST_CASE("Test generate_chunks", TEST_GROUP) {
    CATCH_SECTION("Invalid input") {
        // num_samples == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(0, 9996, 6, 498));
        // chunk_size == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 0, 6, 498));
        // stride == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 9996, 0, 498));
        // (chunk_size % stride) != 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 9996, 10, 498));
        // (overlap % stride) != 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 9996, 7, 498));
        // chunk_size <= overlap
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 9996, 6, 9996));
        CATCH_REQUIRE_THROWS(dorado::utils::generate_chunks(12345, 9996, 6, 9997));
    }

    CATCH_SECTION("Valid input") {
        CATCH_REQUIRE(dorado::utils::generate_chunks(9996 / 2, 9996, 6, 498) ==
                      std::vector<std::size_t>{0});
        CATCH_REQUIRE(dorado::utils::generate_chunks(9996, 9996, 6, 498) ==
                      std::vector<std::size_t>{0});
        CATCH_REQUIRE(dorado::utils::generate_chunks(9996 + 1, 9996, 6, 498) ==
                      std::vector<std::size_t>{0, 6});
        CATCH_REQUIRE(dorado::utils::generate_chunks(9996 + (9996 / 2), 9996, 6, 498) ==
                      std::vector<std::size_t>{0, 4998});
        CATCH_REQUIRE(dorado::utils::generate_chunks((2 * 9996) + (9996 / 2), 9996, 1, 0) ==
                      std::vector<std::size_t>{0, 9996, 14994});
        CATCH_REQUIRE(dorado::utils::generate_chunks(3 * 9996, 9996, 6, 498) ==
                      std::vector<std::size_t>{0, 9498, 18996, 19992});

        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(1024, 2097152);

        const auto validate_chunks = [&](const std::size_t chunk_size, const std::size_t stride,
                                         const std::size_t overlap) {
            std::vector<std::size_t> reads;
            reads.reserve(16);
            for (int i = 0; i < 16; ++i) {
                reads.emplace_back(distribution(generator));
            }
            for (const std::size_t num_samples : reads) {
                std::vector<std::size_t> offsets;
                CATCH_REQUIRE_NOTHROW(offsets = dorado::utils::generate_chunks(
                                              num_samples, chunk_size, stride, overlap));
                CATCH_REQUIRE_FALSE(std::empty(offsets));

                CATCH_REQUIRE(offsets.front() == 0);

                for (std::size_t i = 1; i < (std::size(offsets) - 1); ++i) {
                    CATCH_REQUIRE((offsets[i] % stride) == 0);
                    CATCH_REQUIRE(offsets[i] == (i * (chunk_size - overlap)));
                }

                CATCH_REQUIRE((offsets.back() % stride) == 0);
                CATCH_REQUIRE(offsets.back() < num_samples);
                if (std::size(offsets) > 1) {
                    CATCH_REQUIRE((num_samples - offsets.back()) >= (chunk_size - stride));
                    CATCH_REQUIRE((num_samples - offsets.back()) <= chunk_size);
                }
            }
        };

        validate_chunks(9996, 6, 498);
        validate_chunks(9996, 7, 497);
        validate_chunks(9996, 12, 492);
        validate_chunks(9996, 17, 510);
        validate_chunks(555, 5, 25);
        validate_chunks(83, 1, 13);
        validate_chunks(123, 1, 0);
    }
}

CATCH_TEST_CASE("Test generate_variable_chunks", TEST_GROUP) {
    CATCH_SECTION("Invalid input") {
        // num_samples == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(0, 9996, 6, 498));
        // chunk_size == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 0, 6, 498));
        // stride == 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 0, 498));
        // (chunk_size % stride) != 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 10, 498));
        // chunk_size == stride
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 6, 6, 498));
        // (overlap % stride) != 0
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 7, 498));
        // (stride != 1) && (overlap == 0)
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 7, 0));
        // chunk_size <= overlap
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 6, 9996));
        CATCH_REQUIRE_THROWS(dorado::utils::generate_variable_chunks(12345, 9996, 6, 9997));
    }

    CATCH_SECTION("Valid input") {
        using Interval = std::pair<std::size_t, std::size_t>;
        CATCH_REQUIRE(dorado::utils::generate_variable_chunks(9996 / 2, 9996, 6, 498) ==
                      std::vector<Interval>{{0, 4998}});
        CATCH_REQUIRE(dorado::utils::generate_variable_chunks(9996, 9996, 6, 498) ==
                      std::vector<Interval>{{0, 9996}});
        CATCH_REQUIRE(dorado::utils::generate_variable_chunks(9996 + 1, 9996, 6, 498) ==
                      std::vector<Interval>{{0, 5244}, {4752, 9997}});
        CATCH_REQUIRE(dorado::utils::generate_variable_chunks(9996 + (9996 / 2), 9996, 6, 498) ==
                      std::vector<Interval>{{0, 7746}, {7248, 14994}});
        CATCH_REQUIRE(
                dorado::utils::generate_variable_chunks((2 * 9996) + (9996 / 2), 9996, 1, 0) ==
                std::vector<Interval>{{0, 8330}, {8330, 16660}, {16660, 24990}});
        CATCH_REQUIRE(
                dorado::utils::generate_variable_chunks(3 * 9996, 9996, 6, 498) ==
                std::vector<Interval>{{0, 7866}, {7374, 15240}, {14748, 22614}, {22122, 29988}});

        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(1024, 2097152);

        const auto validate_chunks = [&](const std::size_t chunk_size, const std::size_t stride,
                                         const std::size_t overlap) {
            std::vector<std::size_t> reads;
            reads.reserve(16);
            for (int i = 0; i < 16; ++i) {
                reads.emplace_back(distribution(generator));
            }
            for (const std::size_t num_samples : reads) {
                std::vector<Interval> intervals;
                CATCH_REQUIRE_NOTHROW(intervals = dorado::utils::generate_variable_chunks(
                                              num_samples, chunk_size, stride, overlap));
                CATCH_REQUIRE_FALSE(std::empty(intervals));

                CATCH_REQUIRE(intervals.front().first == 0);
                for (std::size_t i = 1; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i].first % stride) == 0);
                }

                for (std::size_t i = 0; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i].second - intervals[i].first) > 0);
                    CATCH_REQUIRE((intervals[i].second - intervals[i].first) <= chunk_size);
                }
                for (std::size_t i = 1; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i - 1].second - intervals[i].first) <= overlap);
                }

                for (std::size_t i = 0; i < (std::size(intervals) - 1); ++i) {
                    CATCH_REQUIRE((intervals[i].second % stride) == 0);
                }
                CATCH_REQUIRE(intervals.back().second == num_samples);
            }
        };

        validate_chunks(9996, 6, 498);
        validate_chunks(9996, 7, 497);
        validate_chunks(9996, 12, 492);
        validate_chunks(9996, 17, 510);
        validate_chunks(555, 5, 25);
        validate_chunks(83, 1, 13);
        validate_chunks(123, 1, 0);
    }
}