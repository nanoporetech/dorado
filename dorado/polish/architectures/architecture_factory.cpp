#include "architecture_factory.h"

#include "counts_feature_encoder.h"
#include "gru_model.h"
#include "torch_script_model.h"

namespace dorado::polisher {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

LabelSchemeType parse_label_scheme_type(const std::string& type) {
    if (type == "HaploidLabelScheme") {
        return LabelSchemeType::HAPLOID;
    }
    throw std::runtime_error{"Unknown label scheme type: '" + type + "'!"};
}

FeatureEncoderType parse_feature_encoder_type(const std::string& type) {
    if (type == "CountsFeatureEncoder") {
        return FeatureEncoderType::COUNTS_FEATURE_ENCODER;
    }
    throw std::runtime_error{"Unknown feature encoder type: '" + type + "'!"};
}

// std::unique_ptr<TorchModel> create_model(const std::filesystem::path& model_path,
//                                                    const DeviceInfo& device_info,
//                                                    const bool full_precision) {
//     // Load weights from the model file.
//     torch::jit::script::Module module;

//     try {
//         spdlog::debug("Loading weights from file: {}", model_path.string());
//         module = torch::jit::load(model_path.string());
//     } catch (const c10::Error& e) {
//         throw std::runtime_error("Error loading model from " + model_path.string() +
//                                  " with error: " + e.what());
//     }

//     // Construct the model.
//     spdlog::debug("Creating the GRU model.");
//     std::unique_ptr<GRUModel> model = std::make_unique<GRUModel>(10, 5, 128);

//     spdlog::debug("Setting the weights.");
//     auto state_dict = module.named_parameters();
//     for (const auto& p : state_dict) {
//         auto* param = model->named_parameters().find(p.name);
//         if (param != nullptr) {
//             param->copy_(p.value);
//         } else {
//             throw std::runtime_error(
//                     "Some loaded parameters cannot be found in the C++ model! name = " + p.name);
//         }
//     }
//     model->to(device_info.device);
//     if ((device_info.type == DeviceType::CUDA) && !full_precision) {
//         model->to_half();
//         spdlog::info("Converted the model to half.");
//     } else {
//         spdlog::info("Using full precision.");
//     }
//     model->eval();

//     size_t total_params = 0;
//     size_t total_bytes = 0;
//     for (const auto& param : model->parameters()) {
//         total_params += param.numel();
//         total_bytes += param.numel() * param.element_size();
//     }
//     spdlog::info("Model: total parameters: {}, size: {} MB", total_params,
//                  (total_bytes / (1024.0 * 1024.0)));

//     return model;
// }

// template<typename T>
// T parse_config_kwarg(const std::unordered_map<std::string, std::string>& dict, const std::string& key) {

// }

PolishArchitecture architecture_factory(const ModelConfig& config) {
    // const auto create_models = []() {
    //     std::vector<std::shared_ptr<TorchModel>> ret;
    //     for (int32_t device_id = 0; device_id < dorado::ssize(devices); ++device_id) {
    //         ret.emplace_back(create_model(opt.model_path / "model.pt", devices[device_id],
    //                                       opt.full_precision));
    //         spdlog::info("Loaded model to device {}: {}", device_id, devices[device_id].name);
    //     }
    //     if ((std::size(devices) == 1) && (devices.front().type == DeviceType::CPU)) {
    //         for (int32_t i = 1; i < opt.threads; ++i) {
    //             ret.emplace_back(ret.front());
    //         }
    //     }
    //     return ret;
    // };
    // const std::vector<std::shared_ptr<TorchModel>> models = create_models();

    const auto get_value = [](const std::unordered_map<std::string, std::string>& dict,
                              const std::string& key) -> std::string {
        const auto it = dict.find(key);
        if (it == std::cend(dict)) {
            throw std::runtime_error{"Cannot find key '" + key + "' in kwargs!"};
        }
        return it->second;
    };

    PolishArchitecture arch;

    arch.model_type = parse_model_type(config.model_type);
    arch.label_scheme_type = parse_label_scheme_type(config.label_scheme_type);
    arch.feature_encoder_type = parse_feature_encoder_type(config.feature_encoder_type);

    // std::unique_ptr<TorchModel> model = create_model();

    std::unique_ptr<TorchModel> model;

    if (config.model_file == "model.pt") {
        // Load a TorchScript model. Parameters are not important here.
        model = std::make_unique<TorchScriptModel>(config.model_dir / config.model_file);

    } else if (arch.model_type == ModelType::GRU) {
        const int32_t num_features = std::atoi(get_value(config.model_kwargs, "num_features"));
        const int32_t num_classes = std::atoi(get_value(config.model_kwargs, "num_classes"));
        const int32_t gru_size = std::atoi(get_value(config.model_kwargs, "gru_size"));
        const bool n_layers = std::atoi(get_value(config.model_kwargs, "n_layers"));
        const bool bidirectional = std::atoi(get_value(config.model_kwargs, "bidirectional"));
        constexpr bool NORMALISE = true;
        // const double read_majority_threshold = std::atod(get_value(config.model_kwargs, "read_majority_threshold"));

        model = std::make_unique<GRUModel>(num_features, num_classes, gru_size, n_layers,
                                           bidirectional, NORMALISE);
    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    arch.models.emplace_back(std::move(model));

    return arch;
}

}  // namespace dorado::polisher
