#include "torch_utils/cuda_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <spdlog/spdlog.h>

#include <limits>
#include <tuple>

#define CUT_TAG "[cuda_utils]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace Catch::Matchers;

namespace dorado::utils::cuda_utils {

using details::try_parse_device_ids;

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

    auto options = at::TensorOptions().dtype(torch::kFloat16).device(c10::kCUDA);
    auto A = torch::rand({L, M}, options);
    auto B = torch::rand({M, N}, options);
    auto C1 = torch::empty({L, N}, options);
    auto C2 = torch::empty({L, N}, options);

    // Do it both ways
    details::matmul_f16_cublas(A, B, C1);
    details::matmul_f16_torch(A, B, C2);

    // Compare results
    // Note that half precision floating point only has enough mantissa for
    // ~3 decimal digits, so we need to reduce the tolerances a bit.
    const double rtol = 1e-3;
    const double atol = 0;
    CATCH_REQUIRE(torch::allclose(C1, C2, rtol, atol));
}

DEFINE_TEST("try_parse_device_ids parameterised test cases") {
    auto [device_string, num_devices, expected_result, expected_ids] =
            GENERATE(table<std::string, std::size_t, bool, std::vector<int>>({
                    {"cpu", 0, true, {}},
                    {"cpu", 1, true, {}},
                    {"cuda:all", 1, true, {0}},
                    {"cuda:all", 0, false, {}},
                    {"cuda:all", 4, true, {0, 1, 2, 3}},
                    {"cuda:2", 2, false, {}},
                    {"cuda:-1", 1, false, {}},
                    {"cuda:2", 3, true, {2}},
                    {"cuda:2,0,3", 4, true, {0, 2, 3}},
                    {"cuda:0,0", 4, false, {}},
                    {"cuda:0,1,2,1", 4, false, {}},
                    {"cuda:a", 4, false, {}},
                    {"cuda:a,0", 4, false, {}},
                    {"cuda:0,a", 4, false, {}},
                    {"cuda:1-3", 4, false, {}},
                    {"cuda:1.3", 4, false, {}},
            }));
    CATCH_CAPTURE(device_string);
    CATCH_CAPTURE(num_devices);
    std::vector<int> device_ids{};
    std::string error_message{};

    CATCH_CHECK(try_parse_device_ids(device_string, num_devices, device_ids, error_message) ==
                expected_result);
    CATCH_CHECK_THAT(device_ids, UnorderedEquals(expected_ids));
}

}  // namespace dorado::utils::cuda_utils
