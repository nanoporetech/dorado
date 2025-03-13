#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace dorado::config {

struct ModBaseBatchParams {
    std::size_t batchsize;
    std::size_t runners_per_caller;
    std::size_t threads;
    float threshold;

    std::string to_string() const;
};

ModBaseBatchParams get_modbase_params(const std::vector<std::filesystem::path>& paths);

}  // namespace dorado::config