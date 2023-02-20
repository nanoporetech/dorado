#include "utils/cuda_utils.h"

#include <catch2/catch.hpp>
#include <spdlog/spdlog.h>

#include <limits>
#include <tuple>

#define CUT_TAG "[cuda_utils]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace {

DEFINE_TEST("try_select_max_batch_sizes valid params does not throw") {
    std::vector<int> const breakpoints{1};
    std::vector<std::array<int, 3>> const batch_sizes{{1, 2, 3}};

    REQUIRE_NOTHROW(
            dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0));
}

DEFINE_TEST("try_select_max_batch_sizes empty arrays returns no result") {
    std::vector<int> const breakpoints{};
    std::vector<std::array<int, 3>> const batch_sizes{};

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0);

    REQUIRE_FALSE(result.has_value());
}

DEFINE_TEST("try_select_max_batch_sizes single breakpoint with low value returns no result") {
    std::vector<int> const breakpoints{1};
    std::vector<std::array<int, 3>> const batch_sizes{{1, 2, 3}};

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 0);

    REQUIRE_FALSE(result.has_value());
}

DEFINE_TEST("try_select_max_batch_sizes single breakpoint with high value returns result") {
    std::vector<int> const breakpoints{1};
    std::array<int, 3> max_values{1, 2, 3};
    std::vector<std::array<int, 3>> const batch_sizes{
            max_values,
    };

    auto result = dorado::utils::details::try_select_max_batch_sizes(breakpoints, batch_sizes, 999);

    REQUIRE(result.has_value());
    REQUIRE(*result == max_values);
}

DEFINE_TEST(": try_select_max_batch_sizes single breakpoint with breakpoint value returns result") {
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

DEFINE_TEST("try_select_max_batch_sizes parameterised") {
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

DEFINE_TEST("matmul_f16") {
    // Seed RNG for repeatability in CI
    torch::manual_seed(0);

    // Tensor sizes
    const int L = 3;
    const int M = 4;
    const int N = 5;

    // Setup tensors
    if (!torch::hasCUDA()) {
        spdlog::warn("No Nvidia driver present - Test skipped");
        return;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(c10::kCUDA);
    auto A = torch::rand({L, M}, options);
    auto B = torch::rand({M, N}, options);
    auto C1 = torch::empty({L, N}, options);
    auto C2 = torch::empty({L, N}, options);

    // Do it both ways
    dorado::utils::details::matmul_f16_cublas(A, B, C1);
    dorado::utils::details::matmul_f16_torch(A, B, C2);

    // Compare results
    // Note that half precision floating point only has enough mantissa for
    // ~3 decimal digits, so we need to reduce the tolerances a bit.
    const double rtol = 1e-3;
    const double atol = 0;
    REQUIRE(torch::allclose(C1, C2, rtol, atol));
}

}  // namespace
