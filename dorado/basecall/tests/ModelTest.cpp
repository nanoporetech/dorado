#include "model/Model.h"

#include <ATen/ops/ones.h>
#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[Model]"

namespace {

struct FakeModelImpl : torch::nn::Module {
    at::Tensor forward(const at::Tensor& input,
                       dorado::nn::AuxiliaryData*,
                       at::Tensor* signal_embeddings) {
        if (signal_embeddings != nullptr) {
            *signal_embeddings = input + 1;
        }
        return input * 2;
    }

protected:
    FORWARD_HAS_DEFAULT_ARGS(
            {1, torch::nn::AnyValue(static_cast<dorado::nn::AuxiliaryData*>(nullptr))},
            {2, torch::nn::AnyValue(static_cast<at::Tensor*>(nullptr))});
};

TORCH_MODULE(FakeModel);

dorado::basecall::model::Model make_model() {
    torch::nn::ModuleHolder<torch::nn::AnyModule> module{FakeModel()};
    return dorado::basecall::model::Model(std::move(module));
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " presents a stable output contract", CUT_TAG) {
    auto model = make_model();
    const auto input = at::ones({2, 3});

    const auto output = model.forward(input);

    CATCH_CHECK(output.scores.equal(input * 2));
    CATCH_CHECK_FALSE(output.signal_embeddings.defined());
}

CATCH_TEST_CASE(CUT_TAG " collects optional signal embeddings", CUT_TAG) {
    auto model = make_model();
    const auto input = at::ones({2, 3});

    const auto output = model.forward(input, nullptr, true);

    CATCH_CHECK(output.scores.equal(input * 2));
    CATCH_CHECK(output.signal_embeddings.equal(input + 1));
}
