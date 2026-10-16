#pragma once

#include "config/ModBaseModelConfigFwd.h"

#include <span>
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

ModBaseBatchParams get_modbase_params(std::span<const ModBaseModelConfig> configs,
                                      size_t device_count);

}  // namespace dorado::config