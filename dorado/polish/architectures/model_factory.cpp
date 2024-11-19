#include "model_factory.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <filesystem>
#include <unordered_map>

namespace dorado::polisher {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

std::unique_ptr<TorchModel> model_factory(const ModelConfig& config) {
    const auto get_value = [](const std::unordered_map<std::string, std::string>& dict,
                              const std::string& key) -> std::string {
        const auto it = dict.find(key);
        if (it == std::cend(dict)) {
            throw std::runtime_error{"Cannot find key '" + key + "' in kwargs!"};
        }
        return it->second;
    };

    const ModelType model_type = parse_model_type(config.model_type);

    std::unique_ptr<TorchModel> model;

    if (config.model_file == "model.pt") {
        // Load a TorchScript model. Parameters are not important here.
        spdlog::debug("Loading a TorchScript model.");
        model = std::make_unique<TorchScriptModel>(config.model_dir / config.model_file);

    } else if (model_type == ModelType::GRU) {
        const int32_t num_features = std::stoi(get_value(config.model_kwargs, "num_features"));
        const int32_t num_classes = std::stoi(get_value(config.model_kwargs, "num_classes"));
        const int32_t gru_size = std::stoi(get_value(config.model_kwargs, "gru_size"));
        const bool n_layers = std::stoi(get_value(config.model_kwargs, "n_layers"));
        const bool bidirectional = std::stoi(get_value(config.model_kwargs, "bidirectional"));
        constexpr bool NORMALISE = true;
        // const double read_majority_threshold = std::stod(get_value(config.model_kwargs, "read_majority_threshold"));

        model = std::make_unique<GRUModel>(num_features, num_classes, gru_size, n_layers,
                                           bidirectional, NORMALISE);

    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    // Set the weights of the internally constructed model.
    if (config.model_file != "model.pt") {
        const std::filesystem::path weights_path = config.model_dir / config.model_file;
        torch::jit::script::Module module;
        try {
            spdlog::debug("Loading weights from file: {}", weights_path.string());
            module = torch::jit::load(weights_path.string());
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model weights from " + weights_path.string() +
                                     " with error: " + e.what());
        }
        spdlog::debug("Setting the weights.");
        auto state_dict = module.named_parameters();
        for (const auto& p : state_dict) {
            auto* param = model->named_parameters().find(p.name);
            if (param != nullptr) {
                param->copy_(p.value);
            } else {
                throw std::runtime_error(
                        "Some loaded parameters cannot be found in the libtorch model! name = " +
                        p.name);
            }
        }
    }

    return model;
}

}  // namespace dorado::polisher
