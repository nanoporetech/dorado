#pragma once

#include "nn/ModBaseModelConfig.h"

#include <string>
#include <vector>

namespace dorado {

class MotifMatcher {
public:
    MotifMatcher(const ModBaseModelConfig& model_config);
    std::vector<size_t> get_motif_hits(const std::string& seq) const;

private:
    ModBaseModelConfig m_config;
};

}  // namespace dorado
