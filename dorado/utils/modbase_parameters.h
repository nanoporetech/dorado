#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace dorado::utils::modbase {

enum ModelType { CONV_LSTM_V1, CONV_LSTM_V2, CONV_V1, UNKNOWN };
std::string to_string(const ModelType& model_type) noexcept;
ModelType model_type_from_string(const std::string& model_type) noexcept;
ModelType get_modbase_model_type(const std::filesystem::path& path) noexcept;
bool is_modbase_model(const std::filesystem::path& path);

struct ModBaseParams {
    std::size_t batchsize;
    std::size_t runners_per_caller;
    std::size_t threads;
    float threshold;

    std::string to_string() const;
};

ModBaseParams get_modbase_params(const std::vector<std::filesystem::path>& paths,
                                 int batch_size_arg);

}  // namespace dorado::utils::modbase
