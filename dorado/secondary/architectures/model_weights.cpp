#include "secondary/architectures/model_weights.h"

#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>
#include <torch/serialize.h>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace dorado::secondary {
namespace {

std::vector<char> load_file_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open model weights file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::unordered_set<std::string> tensor_names(
        const torch::OrderedDict<std::string, at::Tensor>& params,
        const torch::OrderedDict<std::string, at::Tensor>& buffers,
        const std::unordered_set<std::string>& non_persistent_buffers) {
    std::unordered_set<std::string> names;
    for (const auto& param : params) {
        names.emplace(param.key());
    }
    for (const auto& buffer : buffers) {
        if (non_persistent_buffers.count(buffer.key()) == 0) {
            names.emplace(buffer.key());
        }
    }
    return names;
}

void validate_loaded_tensors(const std::unordered_set<std::string>& expected,
                             const std::unordered_set<std::string>& loaded) {
    std::vector<std::string> missing;
    for (const std::string& name : expected) {
        if (loaded.count(name) == 0) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error(
                "Cannot load weights into the model: missing tensors in weights "
                "file: " +
                utils::print_container_as_string(missing, ", ", true));
    }

    std::vector<std::string> unexpected;
    for (const std::string& name : loaded) {
        if (expected.count(name) == 0) {
            unexpected.push_back(name);
        }
    }
    if (!unexpected.empty()) {
        throw std::runtime_error(
                "Cannot load weights into the model: unexpected tensors in "
                "weights file: " +
                utils::print_container_as_string(unexpected, ", ", true));
    }
}

void copy_tensor(const std::string& name,
                 const at::Tensor& src,
                 torch::OrderedDict<std::string, at::Tensor>& params,
                 torch::OrderedDict<std::string, at::Tensor>& buffers) {
    if (params.contains(name)) {
        params[name].copy_(src);
    } else if (buffers.contains(name)) {
        buffers[name].copy_(src);
    } else {
        throw std::runtime_error("Unexpected tensor in model weights: " + name);
    }
}

void trace_model_tensors(torch::nn::Module& model,
                         const std::unordered_set<std::string>& non_persistent_buffers) {
    if (!spdlog::default_logger()->should_log(spdlog::level::trace)) {
        return;
    }

    for (const auto& weight : model.named_parameters()) {
        spdlog::trace("[model_params] w.key() = {}", weight.key());
    }
    for (const auto& buffer : model.named_buffers()) {
        spdlog::trace("[model_params] Buffer key: {}, shape: {}, persistent: {}", buffer.key(),
                      (buffer.value().defined() ? utils::tensor_shape_as_string(buffer.value())
                                                : "undefined"),
                      ((non_persistent_buffers.count(buffer.key()) != 0) ? "no" : "yes"));
    }
}

void trace_loaded_tensors(const c10::Dict<c10::IValue, c10::IValue>& weights) {
    if (!spdlog::default_logger()->should_log(spdlog::level::trace)) {
        return;
    }

    for (const auto& weight : weights) {
        spdlog::trace("[loaded pt_param] w.key() = {}", weight.key().toStringRef());
    }
}

}  // namespace

void load_state_dict(torch::nn::Module& model,
                     const std::filesystem::path& weights_path,
                     const std::unordered_set<std::string>& non_persistent_buffers) {
    spdlog::debug("Loading model weights from file: {}", weights_path.string());
    trace_model_tensors(model, non_persistent_buffers);

    try {
        at::InferenceMode infer_guard;

        const std::vector<char> bytes = load_file_bytes(weights_path);
        const c10::Dict<c10::IValue, c10::IValue> weights =
                torch::pickle_load(bytes).toGenericDict();
        trace_loaded_tensors(weights);

        torch::OrderedDict<std::string, at::Tensor> params = model.named_parameters(true);
        torch::OrderedDict<std::string, at::Tensor> buffers = model.named_buffers(true);
        const std::unordered_set<std::string> expected =
                tensor_names(params, buffers, non_persistent_buffers);

        std::unordered_set<std::string> loaded;
        for (const auto& weight : weights) {
            const std::string name = weight.key().toStringRef();
            loaded.emplace(name);
            copy_tensor(name, weight.value().toTensor(), params, buffers);
        }

        validate_loaded_tensors(expected, loaded);
    } catch (const c10::Error& e) {
        throw std::runtime_error{"Error loading model weights from " + weights_path.string() +
                                 ": " + e.what()};
    }
}

}  // namespace dorado::secondary
