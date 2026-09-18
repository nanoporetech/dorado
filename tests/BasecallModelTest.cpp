#include "../dorado/basecall/model/CRFModel.h"
#include "config/BasecallModelConfig.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[basecall_model]"

namespace {

dorado::config::BasecallModelConfig make_flstm_config() {
    dorado::config::BasecallModelConfig config;
    config.convs = {{1, 4, 3, 1, dorado::config::Activation::TANH, 0}};
    config.lstm_size = 4;
    config.lstm_inner_dim = 2;
    config.lstm_layers = 1;
    config.out_features = 3;
    config.outsize = 5;
    config.scale = 5.f;
    return config;
}

}  // namespace

CATCH_TEST_CASE("FLSTM basecaller optionally returns pre-linear signal embeddings", TEST_GROUP) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    dorado::basecall::model::CRFModel model(make_flstm_config(), options);
    for (auto& parameter : model->parameters()) {
        parameter.normal_(0.f, 0.1f);
    }
    model->eval();

    const auto input = torch::randn({2, 1, 8}, options);
    const auto expected_embeddings = model->rnns->forward(model->convs->forward(input));
    const auto expected_scores = model->clamp1->forward(
            model->linear2->forward(model->linear1->forward(expected_embeddings)));

    torch::nn::AnyModule any_model(model);
    at::Tensor embeddings;
    const auto scores =
            any_model.forward(input, static_cast<dorado::nn::AuxiliaryData*>(nullptr), &embeddings);
    const auto scores_without_embeddings = any_model.forward(input);

    CATCH_CHECK(embeddings.defined());
    CATCH_CHECK(embeddings.sizes() == torch::IntArrayRef({2, 8, 4}));
    CATCH_CHECK(at::allclose(embeddings, expected_embeddings));
    CATCH_CHECK(at::allclose(scores, expected_scores));
    CATCH_CHECK(at::allclose(scores, scores_without_embeddings));
}
