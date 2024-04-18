#include <basecall/nn/TxModel.h>
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

// clang-format off
#include <catch2/catch.hpp>
// clang-format on

#include <vector>

#if TORCH_VERSION_MAJOR >= 2
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#define TEST_TAG "[SDPA]"

using namespace dorado::basecall::nn;

TEST_CASE(TEST_TAG " Test Scaled Dot Product Attention", TEST_TAG) {
#if TORCH_VERSION_MAJOR < 2
    spdlog::warn("TORCH_VERSION_MAJOR < 2 - Tests skipped - Scaled Dot Product Attention");
#else
    auto options = at::TensorOptions().dtype(torch::kFloat64).device(c10::kCUDA);

    SECTION("No Mask") {
        torch::manual_seed(0);
        if (!torch::hasCUDA()) {
            spdlog::warn(
                    "No Nvidia driver present - Test skipped - Scaled Dot Product Attention "
                    "[no mask]");
            return;
        }

        torch::Tensor no_mask;
        std::vector<at::Tensor> qkv = torch::rand({8, 8, 8, 3}, options).chunk(3, -1);
        const auto naive_res = scaled_dot_product_attention_naive(qkv[0], qkv[1], qkv[2], no_mask);
        const auto torch_res = at::scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]);
        CHECK(at::allclose(naive_res, torch_res));
    }

    SECTION("Masked") {
        torch::manual_seed(1);
        if (!torch::hasCUDA()) {
            spdlog::warn(
                    "No Nvidia driver present -  Test skipped - Scaled Dot Product Attention "
                    "[mask]");
            return;
        }

        torch::Tensor mask = torch::rand({8, 8}, options).gt(0.5);
        std::vector<at::Tensor> qkv = torch::rand({8, 8, 8, 3}, options).chunk(3, -1);
        const auto naive_res = scaled_dot_product_attention_naive(qkv[0], qkv[1], qkv[2], mask);
        const auto torch_res = at::scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], mask);
        CHECK(at::allclose(naive_res, torch_res));
    }
#endif  // #if TORCH_VERSION_MAJOR < 2
}
