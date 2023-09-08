#pragma once

#include <string>
#include <vector>

namespace dorado {

struct ModBaseModelConfig;
class MotifMatcher {
public:
    MotifMatcher(const ModBaseModelConfig& model_config);
    std::vector<size_t> get_motif_hits(const std::string& seq) const;

private:
    const std::string m_motif;
    const size_t m_motif_offset;
};

}  // namespace dorado
