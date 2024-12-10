#pragma once
#include "utils/dev_utils.h"
#include "utils/parameters.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace dorado::utils::modbase {

enum ModelType { CONV_LSTM_V1, CONV_LSTM_V2, CONV_V1, UNKNOWN };
std::string to_string(const ModelType& model_type);
ModelType model_type_from_string(const std::string& model_type);
ModelType get_modbase_model_type(const std::filesystem::path& path);

struct DefaultModBaseParameters {
#ifdef DORADO_TX2
    int batchsize{128};
    int batchsize_conv_lstm_v2{128};
#else
    int batchsize{1024};
    int batchsize_conv_lstm_v2{2048};
#endif
    int threads{4};
    int threads_conv_lstm_v2{8};

    int runners_per_caller{2};
    int runners_per_caller_conv_lstm_v2{4};

    float methylation_threshold{0.05f};
};

static const DefaultModBaseParameters default_modbase_parameters{};

struct ModBaseParams {
    size_t batchsize{static_cast<size_t>(default_modbase_parameters.batchsize)};
    size_t runners_per_caller{static_cast<size_t>(default_modbase_parameters.runners_per_caller)};
    int threads{default_modbase_parameters.threads};
    float threshold{default_modbase_parameters.methylation_threshold};

    std::string to_string() const {
        std::string str = "ModBaseParams {";
        str += "batchsize:" + std::to_string(batchsize);
        str += ", runners_per_caller:" + std::to_string(runners_per_caller);
        str += ", threads:" + std::to_string(threads);
        str += ", threshold:" + std::to_string(threshold);
        str += "}";
        return str;
    };
};

ModBaseParams get_modbase_params(const std::vector<std::filesystem::path>& paths,
                                 int batch_size_arg,
                                 float threshold_arg);

}  // namespace dorado::utils::modbase