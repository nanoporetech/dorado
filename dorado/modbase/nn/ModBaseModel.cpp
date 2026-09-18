#include "ModBaseModel.h"

#include "ModBaseConvLSTMModel.h"
#include "ModBaseConvLSTMV3Model.h"
#include "ModBaseConvModel.h"
#include "config/ModBaseModelConfig.h"
#include "utils/dev_utils.h"
#include "utils/overloaded.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

namespace {

template <class Model>
torch::nn::ModuleHolder<torch::nn::AnyModule> populate_model(Model&& model,
                                                             const std::filesystem::path& path,
                                                             const at::TensorOptions& options) {
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

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

}  // namespace dorado::modbase
