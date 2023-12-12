#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace dorado::modbase {

struct ModBaseModelConfig;
class MotifMatcher {
public:
    MotifMatcher(const ModBaseModelConfig& model_config);
    MotifMatcher(const std::string& motif, size_t offset);

    std::vector<size_t> get_motif_hits(std::string_view seq) const;

private:
    const std::string m_motif;
    const size_t m_motif_offset;
};

}  // namespace dorado::modbase
