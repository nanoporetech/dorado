#include "utils/cuda_utils.h"

#include <catch2/catch.hpp>
#include <spdlog/spdlog.h>

#include <limits>
#include <tuple>

#define CUT_TAG "[cuda_utils]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

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
    REQUIRE(torch::allclose(C1, C2, rtol, atol));
}

DEFINE_TEST("try_parse_device_ids with non cuda string returns true") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE(try_parse_device_ids("cpu", 1, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:all and 1 device returns true") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE(try_parse_device_ids("cuda:all", 1, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:all and zero devices returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:all", 0, device_ids, error_message));
}

DEFINE_TEST(
        "try_parse_device_ids with cuda:all and num_devices 1 returns device_ids with single entry "
        "'0'") {
    std::vector<int> device_ids{};
    std::string error_message{};

    try_parse_device_ids("cuda:all", 1, device_ids, error_message);

    CHECK(device_ids.size() == 1);
    CHECK(device_ids[0] == 0);
}

DEFINE_TEST(
        "try_parse_device_ids with cuda:all and num_devices 4 returns device_ids with entries "
        "'0,1,2,3'") {
    std::vector<int> device_ids{};
    std::string error_message{};

    try_parse_device_ids("cuda:all", 4, device_ids, error_message);

    std::vector<int> expected_ids{0, 1, 2, 3};
    CHECK_THAT(device_ids, UnorderedEquals(expected_ids));
}

DEFINE_TEST("try_parse_device_ids with cuda:2 and num_devices 2 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    CHECK_FALSE(try_parse_device_ids("cuda:2", 2, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:-1 and num_devices 1 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    CHECK_FALSE(try_parse_device_ids("cuda:-1", 1, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:2 and num_devices 3 returns true") {
    std::vector<int> device_ids{};
    std::string error_message{};

    CHECK(try_parse_device_ids("cuda:2", 3, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:2 and num_devices 3 returns device_ids '2'") {
    std::vector<int> device_ids{};
    std::string error_message{};

    try_parse_device_ids("cuda:2", 3, device_ids, error_message);

    std::vector<int> expected_ids{2};
    CHECK_THAT(device_ids, UnorderedEquals(expected_ids));
}

DEFINE_TEST(
        "try_parse_device_ids with cuda:2,0,3 and num_devices 4 returns device_ids containing "
        "'0,2,3'") {
    std::vector<int> device_ids{};
    std::string error_message{};

    try_parse_device_ids("cuda:2,0,3", 4, device_ids, error_message);

    std::vector<int> expected_ids{0, 2, 3};
    CHECK_THAT(device_ids, UnorderedEquals(expected_ids));
}

DEFINE_TEST("try_parse_device_ids with cuda:0,0 and num_devices 4 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:0,0", 4, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:0,1,2,1 and num_devices 4 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:0,1,2,1", 4, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:a and num_devices 4 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:a", 4, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:a,0 and num_devices 4 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:a,0", 4, device_ids, error_message));
}

DEFINE_TEST("try_parse_device_ids with cuda:0,a and num_devices 4 returns false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:0,a", 4, device_ids, error_message));
}

DEFINE_TEST(
        "try_parse_device_ids with unsupported range syntax cuda:1-3 and num_devices 4 returns "
        "false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:1-3", 4, device_ids, error_message));
}

DEFINE_TEST(
        "try_parse_device_ids with float device id cuda:1.2 and num_devices 4 returns "
        "false") {
    std::vector<int> device_ids{};
    std::string error_message{};

    REQUIRE_FALSE(try_parse_device_ids("cuda:1.2", 4, device_ids, error_message));
}

}  // namespace dorado::utils::cuda_utils
