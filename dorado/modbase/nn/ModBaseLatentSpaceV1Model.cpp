#include "ModBaseLatentSpaceV1Model.h"

#include "torch_utils/module_utils.h"

#include <algorithm>
#include <string_view>
#include <tuple>

namespace {

std::string indexed_layer_name(const std::string_view name,
                               const size_t index,
                               const size_t layer_count) {
    if (layer_count == 1) {
        return std::string(name);
    }
    return std::string(name) + std::to_string(index + 1);
}

std::vector<std::string> load_common_weight_names(
        const dorado::config::modbase::LatentSpaceModulesParams& modules) {
    std::vector<std::string> tensors;
    tensors.reserve((modules.sequence_convs.size() * 2) + (modules.projection.size() * 2) +
                    (modules.transformer_encoder.size() * 7));

    for (size_t i = 0; i < modules.sequence_convs.size(); ++i) {
        const auto prefix = "seq_conv" + std::to_string(i + 1);
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    for (size_t i = 0; i < modules.projection.size(); ++i) {
        const auto prefix = indexed_layer_name("proj", i, modules.projection.size());
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    for (size_t i = 0; i < modules.transformer_encoder.size(); ++i) {
        const auto prefix = "transformer" + std::to_string(i + 1);
        tensors.push_back(prefix + ".Wqkv.weight.tensor");
        tensors.push_back(prefix + ".out_proj.weight.tensor");
        tensors.push_back(prefix + ".out_proj.bias.tensor");
        tensors.push_back(prefix + ".fc1.weight.tensor");
        tensors.push_back(prefix + ".fc2.weight.tensor");
        tensors.push_back(prefix + ".norm1.weight.tensor");
        tensors.push_back(prefix + ".norm2.weight.tensor");
    }
    return tensors;
}

std::vector<std::string> load_classifier_weight_names(
        const dorado::config::modbase::LatentSpaceModulesParams& modules) {
    std::vector<std::string> tensors;
    tensors.reserve(modules.classifier.size() * 2);
    for (size_t i = 0; i < modules.classifier.size(); ++i) {
        const auto prefix = indexed_layer_name("fc", i, modules.classifier.size());
        tensors.push_back(prefix + ".weight.tensor");
        tensors.push_back(prefix + ".bias.tensor");
    }
    return tensors;
}

bool conv_params_equal(const dorado::config::ConvParams& lhs,
                       const dorado::config::ConvParams& rhs) {
    return std::tie(lhs.insize, lhs.size, lhs.winlen, lhs.stride, lhs.activation, lhs.inner_dim) ==
           std::tie(rhs.insize, rhs.size, rhs.winlen, rhs.stride, rhs.activation, rhs.inner_dim);
}

bool linear_params_equal(const dorado::config::modbase::LinearParams& lhs,
                         const dorado::config::modbase::LinearParams& rhs) {
    return std::tie(lhs.in_size, lhs.out_size) == std::tie(rhs.in_size, rhs.out_size);
}

bool transformer_params_equal(const dorado::config::TransformerEncoderLayerParams& lhs,
                              const dorado::config::TransformerEncoderLayerParams& rhs) {
    return std::tie(lhs.d_model, lhs.nhead, lhs.dim_feedforward, lhs.attn_window,
                    lhs.deepnorm_alpha, lhs.theta, lhs.max_seq_len) ==
           std::tie(rhs.d_model, rhs.nhead, rhs.dim_feedforward, rhs.attn_window,
                    rhs.deepnorm_alpha, rhs.theta, rhs.max_seq_len);
}

template <typename T, typename Compare>
bool vectors_equal(const std::vector<T>& lhs, const std::vector<T>& rhs, Compare compare) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(), compare);
}

bool common_configs_equal(const dorado::config::modbase::LatentSpaceV1ModelConfig& lhs,
                          const dorado::config::modbase::LatentSpaceV1ModelConfig& rhs) {
    // num_out and the classifier are intentionally excluded: they belong to the final model.
    return (std::tie(lhs.general.size, lhs.general.kmer_len, lhs.general.stride,
                     lhs.chunked.chunk_size, lhs.sequence_stride) ==
            std::tie(rhs.general.size, rhs.general.kmer_len, rhs.general.stride,
                     rhs.chunked.chunk_size, rhs.sequence_stride)) &&
           vectors_equal(lhs.modules.sequence_convs, rhs.modules.sequence_convs,
                         conv_params_equal) &&
           vectors_equal(lhs.modules.projection, rhs.modules.projection, linear_params_equal) &&
           vectors_equal(lhs.modules.transformer_encoder, rhs.modules.transformer_encoder,
                         transformer_params_equal);
}

bool model_parameters_equal(const torch::nn::Module& lhs, const torch::nn::Module& rhs) {
    const auto lhs_params = lhs.parameters();
    const auto rhs_params = rhs.parameters();
    if (lhs_params.size() != rhs_params.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs_params.size(); ++i) {
        const auto& lhs_param = lhs_params.at(i);
        const auto& rhs_param = rhs_params.at(i);
        if (lhs_param.sizes() != rhs_param.sizes() ||
            lhs_param.scalar_type() != rhs_param.scalar_type()) {
            return false;
        }
        const auto lhs_cpu = lhs_param.device().is_cpu() ? lhs_param : lhs_param.cpu();
        const auto rhs_cpu = rhs_param.device().is_cpu() ? rhs_param : rhs_param.cpu();
        if (!lhs_cpu.equal(rhs_cpu)) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace dorado::modbase::model {

ModBaseLatentSpaceV1CommonModelImpl::ModBaseLatentSpaceV1CommonModelImpl(
        const config::modbase::LatentSpaceV1ModelConfig& config,
        const at::TensorOptions& options)
        : model_config(config) {
    sequence_encoder =
            register_module("sequence_encoder", nn::ConvStack(model_config.modules.sequence_convs));
    projection = register_module("projection", torch::nn::Sequential());
    for (const auto& linear : model_config.modules.projection) {
        projection->push_back(torch::nn::Linear(linear.in_size, linear.out_size));
    }

    common_encoder = register_module("common_encoder", torch::nn::Sequential());
    for (auto params : model_config.modules.transformer_encoder) {
        if (params.max_seq_len == 0) {
            params.max_seq_len =
                    static_cast<int>(model_config.chunked.chunk_size / model_config.general.stride);
        }
        common_encoder->push_back(nn::TxEncoder(params, options, false));
    }
}

at::Tensor ModBaseLatentSpaceV1CommonModelImpl::forward(at::Tensor sequences,
                                                        at::Tensor embedded_features) {
    const auto conv_dtype = (sequences.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;
    // NTC -> NCT -> NTC
    sequences = sequence_encoder->forward(sequences.permute({0, 2, 1}).to(conv_dtype));
    // sequences = sequences.permute({0, 2, 1});
    embedded_features = projection->forward(embedded_features);
    return common_encoder->forward(sequences + embedded_features);
}

void ModBaseLatentSpaceV1CommonModelImpl::load_state_dict(const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseLatentSpaceV1CommonModelImpl::load_weights(
        const std::filesystem::path& dir) const {
    return utils::load_tensors(dir, load_common_weight_names(model_config.modules));
}

bool ModBaseLatentSpaceV1CommonModelImpl::is_compatible(
        const ModBaseLatentSpaceV1CommonModelImpl& other) const {
    return common_configs_equal(model_config, other.model_config) &&
           model_parameters_equal(*this, other);
}

bool ModBaseLatentSpaceV1CommonModelImpl::operator==(
        const ModBaseLatentSpaceV1CommonModelImpl& other) const {
    return is_compatible(other);
}

ModBaseLatentSpaceV1ClassifierModelImpl::ModBaseLatentSpaceV1ClassifierModelImpl(
        const config::modbase::LatentSpaceV1ModelConfig& config)
        : model_config(config) {
    classifier_head = register_module("classifier_head", torch::nn::Sequential());
    for (const auto& linear : model_config.modules.classifier) {
        classifier_head->push_back(torch::nn::Linear(linear.in_size, linear.out_size));
    }
}

at::Tensor ModBaseLatentSpaceV1ClassifierModelImpl::forward(const at::Tensor& common_output) {
    return classifier_head->forward(common_output).permute({0, 2, 1});
}

void ModBaseLatentSpaceV1ClassifierModelImpl::load_state_dict(
        const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseLatentSpaceV1ClassifierModelImpl::load_weights(
        const std::filesystem::path& dir) const {
    return utils::load_tensors(dir, load_classifier_weight_names(model_config.modules));
}

}  // namespace dorado::modbase::model
