
#include "ModBaseBatchParams.h"

#include "ModBaseModelConfig.h"
#include "utils/dev_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace dorado::config {

namespace {

namespace DefaultModBaseParameters {
constexpr std::size_t batchsize{1024};

#if DORADO_TX2
constexpr std::size_t batchsize_conv_lstm_v2{128};
#else
constexpr std::size_t batchsize_conv_lstm_v2{2048};
#endif

constexpr std::size_t threads{4};
constexpr std::size_t threads_conv_lstm_v2{8};

constexpr std::size_t runners_per_caller{2};
constexpr std::size_t runners_per_caller_conv_lstm_v2{4};

constexpr float methylation_threshold{0.05f};
};  // namespace DefaultModBaseParameters

}  // namespace

std::string ModBaseBatchParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "ModBaseBatchParams {"
        << " batchsize:" << batchsize
        << " runners_per_caller:" << runners_per_caller
        << " threads:" << threads
        << " threshold:" << threshold << " }";
    return oss.str();
    // clang-format on
}

ModBaseBatchParams get_modbase_params(const std::vector<std::filesystem::path>& paths) {
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
    if (runners_per_caller <= 0 || threads <= 0) {
        throw std::runtime_error("Modbase parameters must be positive integers.");
    }
    return ModBaseBatchParams{batch_size, runners_per_caller, threads,
                              DefaultModBaseParameters::methylation_threshold};
}

}  // namespace dorado::config
