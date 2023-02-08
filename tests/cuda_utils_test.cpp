#include "utils/cuda_utils.h"

#include <catch2/catch.hpp>

#include <limits>

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

const std::vector<std::array<int, 3>> batch_sizes = {batch_sizes_low, batch_sizes_mid,
                                                     batch_sizes_high};

struct BatchSelecterTestCase {
    int available_memory;
    bool success;
    std::array<int, 3> result;
};

std::vector<BatchSelecterTestCase> test_cases{
        {0, false, {}},
        {1, false, {}},
        {2, true, batch_sizes_low},
        {3, true, batch_sizes_low},
        {4, true, batch_sizes_mid},
        {5, true, batch_sizes_mid},
        {6, true, batch_sizes_high},
        {7, true, batch_sizes_high},
        {-1, false, {}},
        {std::numeric_limits<int>::max(), true, batch_sizes_high},
        {std::numeric_limits<int>::min(), false, {}},
};

TEST_CASE(CUT_TAG ": try_select_max_batch_sizes parameterised", CUT_TAG) {
    auto index = GENERATE(range(0, static_cast<int>(test_cases.size())));
    auto const& test_case = test_cases[index];
    INFO("available memory: " + std::to_string(test_case.available_memory));

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes,
                                                                     test_case.available_memory);

    REQUIRE(result.has_value() == test_case.success);
    if (result) {
        REQUIRE(*result == test_case.result);
    }
}

}  // namespace