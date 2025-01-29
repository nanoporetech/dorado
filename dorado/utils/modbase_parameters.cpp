
#include "modbase_parameters.h"

#include "utils/dev_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace dorado::utils::modbase {

namespace {

namespace DefaultModBaseParameters {
#if DORADO_TX2
constexpr std::size_t batchsize{128};
constexpr std::size_t batchsize_conv_lstm_v2{128};
#else
constexpr std::size_t batchsize{1024};
constexpr std::size_t batchsize_conv_lstm_v2{2048};
#endif

constexpr std::size_t threads{4};
constexpr std::size_t threads_conv_lstm_v2{8};

constexpr std::size_t runners_per_caller{2};
constexpr std::size_t runners_per_caller_conv_lstm_v2{4};

constexpr float methylation_threshold{0.05f};
};  // namespace DefaultModBaseParameters

}  // namespace

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

std::string ModBaseParams::to_string() const {
    std::string str = "ModBaseParams {";
    str += "batchsize:" + std::to_string(batchsize);
    str += ", runners_per_caller:" + std::to_string(runners_per_caller);
    str += ", threads:" + std::to_string(threads);
    str += ", threshold:" + std::to_string(threshold);
    str += "}";
    return str;
}

ModBaseParams get_modbase_params(const std::vector<std::filesystem::path>& paths) {
    const bool is_conv_lstm_v2 =
            !paths.empty() && get_modbase_model_type(paths.front()) == ModelType::CONV_LSTM_V2;
    const std::size_t batch_size = is_conv_lstm_v2
                                           ? DefaultModBaseParameters::batchsize_conv_lstm_v2
                                           : DefaultModBaseParameters::batchsize;

    const std::size_t threads = utils::get_dev_opt(
            "modbase_threads", is_conv_lstm_v2 ? DefaultModBaseParameters::threads_conv_lstm_v2
                                               : DefaultModBaseParameters::threads);
    const std::size_t runners_per_caller = utils::get_dev_opt(
            "modbase_runners", is_conv_lstm_v2
                                       ? DefaultModBaseParameters::runners_per_caller_conv_lstm_v2
                                       : DefaultModBaseParameters::runners_per_caller);
    if (batch_size < 0 || runners_per_caller < 0 || threads < 0) {
        throw std::runtime_error("Modbase parameters must be positive integers.");
    }
    return ModBaseParams{batch_size, runners_per_caller, threads,
                         DefaultModBaseParameters::methylation_threshold};
}

}  // namespace dorado::utils::modbase
