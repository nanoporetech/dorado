
#include "modbase_parameters.h"

#include "utils/dev_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <toml/get.hpp>

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace dorado::utils::modbase {

std::string to_string(const ModelType& model_type) noexcept {
    switch (model_type) {
    case CONV_LSTM_V1:
        return std::string("conv_lstm");
    case CONV_LSTM_V2:
        return std::string("conv_lstm_v2");
    case CONV_V1:
        return std::string("conv_v1");
    default:
        return std::string("__UNKNOWN__");
    }
};

ModelType model_type_from_string(const std::string& model_type) noexcept {
    if (model_type == "conv_lstm") {
        return ModelType::CONV_LSTM_V1;
    }
    if (model_type == "conv_lstm_v2") {
        return ModelType::CONV_LSTM_V2;
    }
    if (model_type == "conv_only" || model_type == "conv_v1") {
        return ModelType::CONV_V1;
    }
    return ModelType::UNKNOWN;
}

ModelType get_modbase_model_type(const std::filesystem::path& path) noexcept {
    try {
        const auto config_toml = toml::parse(path / "config.toml");
        if (!config_toml.contains("general")) {
            return ModelType::UNKNOWN;
        }
        return model_type_from_string(toml::find<std::string>(config_toml, "general", "model"));
    } catch (std::exception& e) {
        spdlog::trace("get_modbase_model_type caught exception: {}", e.what());
        return ModelType::UNKNOWN;
    }
}

bool is_modbase_model(const std::filesystem::path& path) {
    return get_modbase_model_type(path) != ModelType::UNKNOWN;
}

ModBaseParams get_modbase_params(const std::vector<std::filesystem::path>& paths,
                                 int batch_size_arg,
                                 float threshold_arg) {
    bool is_conv_lstm_v2 =
            !paths.empty() && get_modbase_model_type(paths.front()) == ModelType::CONV_LSTM_V2;
    int batch_size = is_conv_lstm_v2 ? default_modbase_parameters.batchsize_conv_lstm_v2
                                     : default_modbase_parameters.batchsize;
    if (batch_size_arg > 0) {
        batch_size = batch_size_arg;
    }
    const int threads = utils::get_dev_opt<int>(
            "modbase_threads", is_conv_lstm_v2 ? default_modbase_parameters.threads_conv_lstm_v2
                                               : default_modbase_parameters.threads);
    const int runners_per_caller = utils::get_dev_opt<int>(
            "modbase_runners", is_conv_lstm_v2
                                       ? default_modbase_parameters.runners_per_caller_conv_lstm_v2
                                       : default_modbase_parameters.runners_per_caller);
    if (batch_size < 0 || runners_per_caller < 0 || threads < 0 || threshold_arg < 0) {
        throw std::runtime_error("Modbase CLI parameters must be positive integers.");
    }
    return ModBaseParams{static_cast<size_t>(batch_size), static_cast<size_t>(runners_per_caller),
                         threads, threshold_arg};
}

}  // namespace dorado::utils::modbase