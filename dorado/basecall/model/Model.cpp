#include "Model.h"

#include "basecall/crf_utils.h"

namespace dorado::basecall::model {

Model::Model(torch::nn::ModuleHolder<torch::nn::AnyModule> module) : m_module(std::move(module)) {}

ModelOutput Model::forward(const at::Tensor& input,
                           nn::AuxiliaryData* aux,
                           bool collect_signal_embeddings) {
    ModelOutput output;
    output.scores = collect_signal_embeddings
                            ? m_module->forward(input, aux, &output.signal_embeddings)
                            : m_module->forward(input, aux);
    return output;
}

std::unique_ptr<Model> load_model(const config::BasecallModelConfig& model_config,
                                  const at::TensorOptions& options) {
    return std::make_unique<Model>(load_crf_model(model_config, options));
}

}  // namespace dorado::basecall::model
