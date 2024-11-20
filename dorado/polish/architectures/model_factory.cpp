#include "model_factory.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace dorado::polisher {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

/**
 * \brief This function is a workaround around missing features in torchlib. There
 *          is currently no way to load only the state dict without the model either
 *          being traced of scripted.
 *          Issue: "PytorchStreamReader failed locating file constants.pkl: file not found"
 *          Source: https://github.com/pytorch/pytorch/issues/36577#issuecomment-1279666295
 */
void load_parameters(TorchModel& model, const std::filesystem::path& in_pt) {
    const auto get_bytes = [](const std::filesystem::path& filename) {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                (std::istreambuf_iterator<char>()));
        return bytes;
    };

    torch::NoGradGuard no_grad;

    const std::vector<char> bytes = get_bytes(in_pt);

    const c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(bytes).toGenericDict();

    if (spdlog::default_logger()->level() == spdlog::level::debug) {
        const torch::OrderedDict<std::string, at::Tensor> model_params = model.named_parameters();
        for (const auto& w : model_params) {
            spdlog::debug("[model_params] w.key() = {}", w.key());
        }
    }

    // auto state_dict = model.named_parameters();
    for (const auto& w : weights) {
        const std::string name = w.key().toStringRef();
        const at::Tensor value = w.value().toTensor();
        auto* param = model.named_parameters().find(name);
        if (param != nullptr) {
            param->copy_(value);
        } else {
            throw std::runtime_error(
                    "Some loaded parameters cannot be found in the libtorch model! name = " + name);
        }
    }
}

std::shared_ptr<TorchModel> model_factory(const ModelConfig& config) {
    const auto get_value = [](const std::unordered_map<std::string, std::string>& dict,
                              const std::string& key) -> std::string {
        const auto it = dict.find(key);
        if (it == std::cend(dict)) {
            throw std::runtime_error{"Cannot find key '" + key + "' in kwargs!"};
        }
        if ((std::size(it->second) >= 2) && (it->second.front() == '"') &&
            (it->second.back() == '"')) {
            return it->second.substr(1, std::size(it->second) - 2);
        }
        return it->second;
    };

    const ModelType model_type = parse_model_type(config.model_type);

    std::shared_ptr<TorchModel> model;

    if ((config.model_file != "model.pt") && (config.model_file != "weights.pt")) {
        throw std::runtime_error{"Unexpected weights/model file name! model_file = '" +
                                 config.model_file.string() +
                                 "', expected either 'model.pt' or 'weights.pt'."};
    }

    if (config.model_file == "model.pt") {
        // Load a TorchScript model. Parameters are not important here.
        spdlog::debug("Loading a TorchScript model.");
        model = std::make_unique<TorchScriptModel>(config.model_dir / config.model_file);

    } else if (model_type == ModelType::GRU) {
        spdlog::debug("Constructing a GRU model.");

        const int32_t num_features = std::stoi(get_value(config.model_kwargs, "num_features"));
        const int32_t num_classes = std::stoi(get_value(config.model_kwargs, "num_classes"));
        const int32_t gru_size = std::stoi(get_value(config.model_kwargs, "gru_size"));
        const int32_t n_layers = std::stoi(get_value(config.model_kwargs, "n_layers"));
        const bool bidirectional =
                (get_value(config.model_kwargs, "bidirectional") == "true") ? true : false;
        constexpr bool NORMALISE = true;
        // const double read_majority_threshold = std::stod(get_value(config.model_kwargs, "read_majority_threshold"));

        model = std::make_unique<GRUModel>(num_features, num_classes, gru_size, n_layers,
                                           bidirectional, NORMALISE);

        // Set the weights of the internally constructed model.
        load_parameters(*model, config.model_dir / config.model_file);

    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    return model;
}

}  // namespace dorado::polisher
