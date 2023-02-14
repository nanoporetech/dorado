#include "utils/cuda_utils.h"

#include <catch2/catch.hpp>

#include <limits>
#include <tuple>

#define CUT_TAG "[cuda_utils]"

namespace {

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes valid params does not throw", CUT_TAG) {
    std::vector<int> const breakpoints{1};
    std::vector<std::array<int, 3>> const batch_sizes{{1, 2, 3}};

    REQUIRE_NOTHROW(
            dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0));
}

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes empty arrays returns no result", CUT_TAG) {
    std::vector<int> const breakpoints{};
    std::vector<std::array<int, 3>> const batch_sizes{};

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0);

    REQUIRE_FALSE(result.has_value());
}

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes single breakpoint with low value returns no result",
          CUT_TAG) {
    std::vector<int> const breakpoints{1};
    std::vector<std::array<int, 3>> const batch_sizes{{1, 2, 3}};

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0);

    REQUIRE_FALSE(result.has_value());
}

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes single breakpoint with high value returns result",
          CUT_TAG) {
    std::vector<int> const breakpoints{1};
    std::array<int, 3> max_values{1, 2, 3};
    std::vector<std::array<int, 3>> const batch_sizes{
            max_values,
    };

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 999);

    REQUIRE(result.has_value());
    REQUIRE(*result == max_values);
}

TEST_CASE(CUT_TAG
          ": try_select_max_batch_sizes single breakpoint with breakpoint value returns result",
          CUT_TAG) {
    std::vector<int> const breakpoints{1};
    std::array<int, 3> max_values{1, 2, 3};
    std::vector<std::array<int, 3>> const batch_sizes{
            max_values,
    };

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 1);

    REQUIRE(result.has_value());
    REQUIRE(*result == max_values);
}

const std::vector<int> breakpoints{2, 4, 6};
const std::array<int, 3> batch_sizes_low{0, 1, 2};
const std::array<int, 3> batch_sizes_mid{3, 4, 5};
const std::array<int, 3> batch_sizes_high{6, 7, 8};
const std::array<int, 3> batch_sizes_none{0, 0, 0};

const std::vector<std::array<int, 3>> batch_sizes = {batch_sizes_low, batch_sizes_mid,
                                                     batch_sizes_high};

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes parameterised", CUT_TAG) {
    int available_memory;
    bool expected_success;
    std::array<int, 3> expected_result;
    std::tie(available_memory, expected_success, expected_result) =
            GENERATE(table<int, bool, std::array<int, 3>>(
                    {std::make_tuple(0, false, batch_sizes_none),
                     std::make_tuple(1, false, batch_sizes_none),
                     std::make_tuple(2, true, batch_sizes_low),
                     std::make_tuple(3, true, batch_sizes_low),
                     std::make_tuple(4, true, batch_sizes_mid),
                     std::make_tuple(5, true, batch_sizes_mid),
                     std::make_tuple(6, true, batch_sizes_high),
                     std::make_tuple(7, true, batch_sizes_high),
                     std::make_tuple(-1, false, batch_sizes_none),
                     std::make_tuple(std::numeric_limits<int>::max(), true, batch_sizes_high),
                     std::make_tuple(std::numeric_limits<int>::min(), false, batch_sizes_none)}));
    INFO("available memory: " + std::to_string(available_memory));

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes,
                                                                     available_memory);

    REQUIRE(result.has_value() == expected_success);
    if (result) {
        REQUIRE(*result == expected_result);
    }
}

}  // namespace