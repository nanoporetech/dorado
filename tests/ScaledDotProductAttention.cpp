#include <basecall/nn/TxModel.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <spdlog/spdlog.h>
#include <torch/nn.h>
#include <torch/version.h>

#include <vector>

#if TORCH_VERSION_MAJOR >= 2
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#define TEST_TAG "[SDPA]"

using namespace dorado::basecall::nn;

CATCH_TEST_CASE(TEST_TAG " Test Scaled Dot Product Attention", TEST_TAG) {
#if TORCH_VERSION_MAJOR < 2
    spdlog::warn("Test skipped - Scaled Dot Product Attention");

#else
    static constexpr c10::DeviceType valid_devices[] = {
#if DORADO_CUDA_BUILD
            c10::kCUDA,
#endif  // DORADO_CUDA_BUILD
#if DORADO_METAL_BUILD
            c10::kMPS,
#endif  // DORADO_METAL_BUILD
            c10::kCPU,
    };
    const auto device_type = GENERATE(
            Catch::Generators::from_range(std::begin(valid_devices), std::end(valid_devices)));
    CATCH_CAPTURE(device_type);

    if ((device_type == c10::kCUDA && !torch::hasCUDA()) ||
        (device_type == c10::kMPS && !torch::hasMPS())) {
        spdlog::warn("Test skipped - Scaled Dot Product Attention: no support for {}",
                     c10::DeviceTypeName(device_type));
        return;
    }

    auto options = at::TensorOptions().dtype(torch::kFloat32).device(device_type);

    CATCH_SECTION("No Mask") {
        torch::manual_seed(0);

        torch::Tensor no_mask;
        std::vector<at::Tensor> qkv = torch::rand({8, 8, 8, 3}, options).chunk(3, -1);
        const auto naive_res = scaled_dot_product_attention_naive(qkv[0], qkv[1], qkv[2], no_mask);
        const auto torch_res = at::scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]);
        CATCH_CHECK(at::allclose(naive_res, torch_res));
    }

    CATCH_SECTION("Masked") {
        torch::manual_seed(123);

        torch::Tensor mask = torch::rand({8, 8}, options).gt(0.5);
        std::vector<at::Tensor> qkv = torch::rand({8, 8, 8, 3}, options).chunk(3, -1);
        const auto naive_res = scaled_dot_product_attention_naive(qkv[0], qkv[1], qkv[2], mask);
        const auto torch_res = at::scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], mask);
        CATCH_CHECK(at::allclose(naive_res, torch_res));
    }
#endif  // #if TORCH_VERSION_MAJOR < 2
}
