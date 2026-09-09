#pragma once

#include "config/ModBaseModelConfigFwd.h"

#include <string>
#include <string_view>
#include <vector>

namespace dorado::modbase {

class MotifMatcher {
public:
    explicit MotifMatcher(const config::ModBaseModelConfig& model_config);
    MotifMatcher(const std::string& motif, size_t offset);

    std::vector<size_t> get_motif_hits(std::string_view seq) const;

private:
    const std::string m_motif;
    const size_t m_motif_offset;
};

}  // namespace dorado::modbase
