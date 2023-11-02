#include "utils/cuda_utils.h"

#include <catch2/catch.hpp>
#include <spdlog/spdlog.h>

#include <limits>
#include <tuple>

#define CUT_TAG "[cuda_utils]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace {

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
