
#include "model_factory.h"

#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>
#include <torch/autograd.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/script.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::secondary {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    } else if (type == "LatentSpaceLSTM") {
        return ModelType::LATENT_SPACE_LSTM;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

namespace {

/**
 * \brief This function is a workaround around missing features in torchlib. There
 *          is currently no way to load only the state dict without the model either
 *          being traced or scripted.
 *          Issue: "PytorchStreamReader failed locating file constants.pkl: file not found"
 *          Source: https://github.com/pytorch/pytorch/issues/36577#issuecomment-1279666295
 */
void load_parameters(ModelTorchBase& model, const std::filesystem::path& in_pt) {
    const auto get_bytes = [](const std::filesystem::path& filename) {
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                (std::istreambuf_iterator<char>()));
        return bytes;
    };

    if (spdlog::default_logger()->level() == spdlog::level::debug) {
        const torch::OrderedDict<std::string, at::Tensor> model_params = model.named_parameters();
        for (const auto& w : model_params) {
            spdlog::debug("[model_params] w.key() = {}", w.key());
        }
        for (const auto& buffer : model.named_buffers()) {
            spdlog::debug("[model_params] Buffer key: {}, shape: {}", buffer.key(),
                          (buffer.value().defined() ? utils::tensor_shape_as_string(buffer.value())
                                                    : "undefined"));
        }
    }

    at::InferenceMode infer_guard;

    const std::vector<char> bytes = get_bytes(in_pt);

    try {
        const c10::Dict<c10::IValue, c10::IValue> weights =
                torch::jit::pickle_load(bytes).toGenericDict();

        auto params = model.named_parameters(true /*recurse*/);
        auto buffers = model.named_buffers(true /*recurse*/);

        // Create a set of model parameters.
        std::unordered_set<std::string> set_model_params;
        for (const auto& param : params) {
            set_model_params.emplace(param.key());
        }
        for (const auto& param : buffers) {
            set_model_params.emplace(param.key());
        }

        // Create a set of loaded parameters.
        std::unordered_set<std::string> set_loaded_params;
        for (const auto& w : weights) {
            const std::string& name = w.key().toStringRef();
            set_loaded_params.emplace(name);
        }

        // Validate that all parameters exist.
        {
            std::vector<std::string> missing;
            for (const auto& name : set_model_params) {
                if (set_loaded_params.count(name) == 0) {
                    missing.emplace_back(name);
                }
            }
            if (!std::empty(missing)) {
                throw std::runtime_error(
                        "Cannot load weights into the model: model contains parameters which are "
                        "not present in the weights file. Missing parameters: " +
                        utils::print_container_as_string(missing, ", "));
            }
        }
        {
            std::vector<std::string> missing;
            for (const auto& name : set_loaded_params) {
                if (set_model_params.count(name) == 0) {
                    missing.emplace_back(name);
                }
            }
            if (!std::empty(missing)) {
                throw std::runtime_error(
                        "Cannot load weights into the model: weights file contains parameters "
                        "which are not present in the model. Missing parameters: " +
                        utils::print_container_as_string(missing, ", "));
            }
        }

        // Set the model weights.
        for (const auto& w : weights) {
            const std::string& name = w.key().toStringRef();
            const at::Tensor& value = w.value().toTensor();

            if (params.contains(name)) {
                params[name].copy_(value);

            } else if (buffers.contains(name)) {
                buffers[name].copy_(value);

            } else {
                throw std::runtime_error(
                        "Some loaded parameters cannot be found in the libtorch model! name = " +
                        name);
            }
        }

    } catch (const c10::Error& e) {
        throw std::runtime_error{std::string("Error: ") + e.what()};
    }
}

}  // namespace

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config) {
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

    std::shared_ptr<ModelTorchBase> model;

    if ((config.model_file != "model.pt") && (config.model_file != "weights.pt")) {
        throw std::runtime_error{"Unexpected weights/model file name! model_file = '" +
                                 config.model_file.string() +
                                 "', expected either 'model.pt' or 'weights.pt'."};
    }

    if (config.model_file == "model.pt") {
        // Load a TorchScript model. Parameters are not important here.
        spdlog::debug("Loading a TorchScript model.");
        model = std::make_unique<ModelTorchScript>(config.model_dir / config.model_file);

    } else if (model_type == ModelType::GRU) {
        spdlog::debug("Constructing a GRU model.");

        const int32_t num_features = std::stoi(get_value(config.model_kwargs, "num_features"));
        const int32_t num_classes = std::stoi(get_value(config.model_kwargs, "num_classes"));
        const int32_t gru_size = std::stoi(get_value(config.model_kwargs, "gru_size"));
        const int32_t n_layers = std::stoi(get_value(config.model_kwargs, "n_layers"));
        const bool bidirectional =
                (get_value(config.model_kwargs, "bidirectional") == "true") ? true : false;

        model = std::make_unique<ModelGRU>(num_features, num_classes, gru_size, n_layers,
                                           bidirectional);

        // Set the weights of the internally constructed model.
        load_parameters(*model, config.model_dir / config.model_file);

    } else if (model_type == ModelType::LATENT_SPACE_LSTM) {
        spdlog::debug("Constructing a LATENT_SPACE_LSTM model.");

        const int32_t num_classes = std::stoi(get_value(config.model_kwargs, "num_classes"));
        const int32_t lstm_size = std::stoi(get_value(config.model_kwargs, "lstm_size"));
        const int32_t cnn_size = std::stoi(get_value(config.model_kwargs, "cnn_size"));
        const std::string pooler_type = get_value(config.model_kwargs, "pooler_type");
        const int32_t bases_alphabet_size =
                std::stoi(get_value(config.model_kwargs, "bases_alphabet_size"));
        const int32_t bases_embedding_size =
                std::stoi(get_value(config.model_kwargs, "bases_embedding_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value(config.model_kwargs, "kernel_sizes"));
        const bool use_dwells =
                (get_value(config.model_kwargs, "use_dwells") == "true") ? true : false;

        // Optionally parse the 'bidirectional' option to support older configs.
        bool bidirectional = true;
        if (config.model_kwargs.find("bidirectional") != std::cend(config.model_kwargs)) {
            bidirectional =
                    (get_value(config.model_kwargs, "bidirectional") == "true") ? true : false;
        }

        model = std::make_unique<ModelLatentSpaceLSTM>(
                num_classes, lstm_size, cnn_size, kernel_sizes, pooler_type, use_dwells,
                bases_alphabet_size, bases_embedding_size, bidirectional);

        load_parameters(*model, config.model_dir / config.model_file);

    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    return model;
}

}  // namespace dorado::secondary
