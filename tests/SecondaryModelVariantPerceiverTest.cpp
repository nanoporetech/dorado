#include "../dorado/secondary/architectures/model_variant_perceiver.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <stdexcept>
#include <vector>

namespace dorado::secondary::tests {

#define TEST_GROUP "[ModelVariantPerceiver]"

namespace {

at::Tensor make_depth_tensor(const std::vector<int64_t>& depths) {
    return at::tensor(depths, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

class TestEmbeddingWrapperImpl : public EmbeddingWrapperImpl {
public:
    TestEmbeddingWrapperImpl(const int64_t max_depth,
                             const int64_t dimension,
                             const at::Tensor& embedding_weight)
            : EmbeddingWrapperImpl(max_depth, dimension) {
        if (!embedding_weight.defined()) {
            throw std::runtime_error{"EmbeddingWrapper test weight tensor must be defined."};
        }
        if (embedding_weight.dim() != 2 || embedding_weight.size(0) != max_depth ||
            embedding_weight.size(1) != dimension) {
            throw std::runtime_error{"EmbeddingWrapper test weight tensor has an invalid shape."};
        }

        torch::NoGradGuard no_grad;
        auto params = named_parameters();
        constexpr auto weight_name = "embedding.weight";
        if (!params.contains(weight_name)) {
            throw std::runtime_error{"EmbeddingWrapper test failed to locate embedding.weight."};
        }
        params[weight_name].copy_(embedding_weight);
    }
};
TORCH_MODULE(TestEmbeddingWrapper);

}  // namespace

CATCH_TEST_CASE("EmbeddingWrapper applies embeddings per-sample depth", TEST_GROUP) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const at::Tensor embedding_weights =
            at::tensor({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f}, options)
                    .reshape({4, 2});
    TestEmbeddingWrapper cut(/*max_depth=*/4, /*dimension=*/2, embedding_weights);

    const at::Tensor x = at::zeros({3, 1, 1, 2}, options);
    const at::Tensor depths = make_depth_tensor({2, 1});
    const at::Tensor batched_output = cut->forward(x.clone(), depths);

    // Equivalent reference path: run each sample separately and concatenate.
    const at::Tensor sample0_output = cut->forward(
            x.narrow(/*dim=*/0, /*start=*/0, /*length=*/2).clone(), make_depth_tensor({2}));
    const at::Tensor sample1_output = cut->forward(
            x.narrow(/*dim=*/0, /*start=*/2, /*length=*/1).clone(), make_depth_tensor({1}));
    const at::Tensor expected = at::cat({sample0_output, sample1_output}, /*dim=*/0);

    CATCH_CHECK(at::allclose(batched_output, expected));
}

CATCH_TEST_CASE("EmbeddingWrapper throws if sample depth exceeds max depth", TEST_GROUP) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    EmbeddingWrapper cut(/*max_depth=*/2, /*dimension=*/2);

    const at::Tensor x = at::zeros({3, 1, 1, 2}, options);
    const at::Tensor depths = make_depth_tensor({3});

    CATCH_CHECK_THROWS_WITH(
            cut->forward(x, depths),
            Catch::Matchers::ContainsSubstring("larger than the maximum embedding depth 2"));
}

CATCH_TEST_CASE("AbsoluteRotaryEmbedding applies frequencies per-sample depth", TEST_GROUP) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    AbsoluteRotaryEmbedding cut(/*dim=*/4, /*theta=*/10000.0f, /*max_read_depth=*/4, options);

    const at::Tensor x =
            at::arange(3 * 2 * 1 * 4, options).reshape({3, 2, 1, 4}) + 1.0f;  // C, T, H, D
    const at::Tensor depths = make_depth_tensor({2, 1});
    const at::Tensor batched_output = cut->forward(x.clone(), depths);

    // Equivalent reference path: run each sample separately and concatenate.
    const at::Tensor sample0_output = cut->forward(
            x.narrow(/*dim=*/0, /*start=*/0, /*length=*/2).clone(), make_depth_tensor({2}));
    const at::Tensor sample1_output = cut->forward(
            x.narrow(/*dim=*/0, /*start=*/2, /*length=*/1).clone(), make_depth_tensor({1}));
    const at::Tensor expected = at::cat({sample0_output, sample1_output}, /*dim=*/0);

    CATCH_CHECK(at::allclose(batched_output, expected));
}

CATCH_TEST_CASE("AbsoluteRotaryEmbedding throws if sample depth exceeds max depth", TEST_GROUP) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    AbsoluteRotaryEmbedding cut(/*dim=*/4, /*theta=*/10000.0f, /*max_read_depth=*/2, options);

    const at::Tensor x = at::ones({3, 2, 1, 4}, options);
    const at::Tensor depths = make_depth_tensor({3});

    CATCH_CHECK_THROWS_WITH(
            cut->forward(x, depths),
            Catch::Matchers::ContainsSubstring("larger than the maximum embedding depth 2"));
}

}  // namespace dorado::secondary::tests
