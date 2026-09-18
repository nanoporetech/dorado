#include "modbase/nn/ModBaseModel.h"

#include "ModBaseConvLSTMModel.h"
#include "ModBaseConvLSTMV3Model.h"
#include "ModBaseConvModel.h"
#include "ModBaseLatentSpaceV1Model.h"
#include "ModBaseLatentSpaceV1SequenceModel.h"
#include "config/ModBaseModelConfig.h"
#include "utils/dev_utils.h"
#include "utils/overloaded.h"

#include <spdlog/spdlog.h>

#include <optional>
#include <stdexcept>
#include <utility>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

namespace {

template <class Model>
void load_model_weights(Model& model, const std::filesystem::path& path) {
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
}

template <class Model>
void configure_model(Model& model, const at::TensorOptions& options) {
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();
}

template <class Model>
torch::nn::ModuleHolder<torch::nn::AnyModule> populate_model(Model&& model,
                                                             const std::filesystem::path& path,
                                                             const at::TensorOptions& options) {
    load_model_weights(model, path);
    configure_model(model, options);

    auto module = torch::nn::AnyModule(std::forward<Model>(model));
    return torch::nn::ModuleHolder<torch::nn::AnyModule>(std::move(module));
}

}  // namespace

namespace dorado::modbase {

torch::nn::ModuleHolder<torch::nn::AnyModule> load_modbase_model(
        const config::ModBaseModelConfig& config,
        const at::TensorOptions& options,
        [[maybe_unused]] const int batchsize) {
    at::InferenceMode guard;
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif
    return std::visit(
            utils::overloaded{
                    [&](const config::modbase::ConvOnlyModelConfig& model_config) {
                        auto model = model::ModBaseConvModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMModelConfig& model_config) {
                        auto model = model::ModBaseConvLSTMModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMV2ModelConfig& model_config) {
                        auto model = model::ModBaseConvLSTMModel(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::ConvLSTMV3ModelConfig& model_config) {
#if DORADO_CUDA_BUILD
                        static bool use_torch = utils::get_dev_opt("modbase_torch", false);
                        if (options.device().is_cuda() && !use_torch) {
                            auto model = model::ModBaseConvLSTMV3CUDAModel(model_config, batchsize);
                            return populate_model(std::move(model), model_config.data.model_path,
                                                  options);
                        }
#endif
                        spdlog::trace("Loading ModBaseConvLSTMV3Model - Torch only");
                        auto model = model::ModBaseConvLSTMV3Model(model_config);
                        return populate_model(std::move(model), model_config.data.model_path,
                                              options);
                    },
                    [&](const config::modbase::LatentSpaceV1ModelConfig&)
                            -> torch::nn::ModuleHolder<torch::nn::AnyModule> {
                        throw std::runtime_error(
                                "Latent space modbase model inference is not implemented");
                    },
            },
            config);
}

std::optional<ModBaseLatentSpaceModels> load_modbase_latent_space_model(
        const std::vector<config::ModBaseModelConfig>& configs,
        const at::TensorOptions& options) {
    if (configs.empty()) {
        return std::nullopt;
    }

    at::InferenceMode guard;
#if DORADO_CUDA_BUILD
    c10::optional<c10::Device> device;
    if (options.device().is_cuda()) {
        device = options.device();
    }
    c10::cuda::OptionalCUDAGuard device_guard(device);
#endif

    std::optional<model::ModBaseLatentSpaceV1SequenceModel> sequence;
    std::optional<model::ModBaseLatentSpaceV1CommonModel> common;
    std::filesystem::path common_model_path;
    std::vector<torch::nn::ModuleHolder<torch::nn::AnyModule>> classifiers;
    classifiers.reserve(configs.size());

    for (const auto& config : configs) {
        std::visit(
                utils::overloaded{
                        [&](const config::modbase::LatentSpaceV1ModelConfig& model_config) {
                            auto candidate_common =
                                    model::ModBaseLatentSpaceV1CommonModel(model_config, options);
                            load_model_weights(candidate_common, model_config.data.model_path);

                            if (common.has_value()) {
                                if (!common.value()->is_compatible(*candidate_common)) {
                                    throw std::runtime_error(
                                            "Latent-space common models are incompatible: " +
                                            common_model_path.string() + " and " +
                                            model_config.data.model_path.string());
                                }
                            } else {
                                sequence = model::ModBaseLatentSpaceV1SequenceModel(model_config);
                                common = std::move(candidate_common);
                                common_model_path = model_config.data.model_path;
                            }

                            auto classifier =
                                    model::ModBaseLatentSpaceV1ClassifierModel(model_config);
                            classifiers.push_back(populate_model(
                                    std::move(classifier), model_config.data.model_path, options));
                        },
                        [](const auto&) {
                            throw std::runtime_error(
                                    "Cannot load a non-latent-space model with the latent-space "
                                    "model loader");
                        },
                },
                config);
    }

    configure_model(sequence.value(), options);
    configure_model(common.value(), options);
    auto sequence_holder = torch::nn::ModuleHolder<torch::nn::AnyModule>(
            torch::nn::AnyModule(std::move(sequence).value()));
    auto common_holder = torch::nn::ModuleHolder<torch::nn::AnyModule>(
            torch::nn::AnyModule(std::move(common).value()));
    return ModBaseLatentSpaceModels{std::move(sequence_holder), std::move(common_holder),
                                    std::move(classifiers)};
}

}  // namespace dorado::modbase
