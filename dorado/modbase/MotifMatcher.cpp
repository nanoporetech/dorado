#include "MotifMatcher.h"

#include <nvtx3/nvtx3.hpp>

namespace dorado {

MotifMatcher::MotifMatcher(const ModBaseModelConfig& model_config) : m_config(model_config) {}

std::vector<size_t> MotifMatcher::get_motif_hits(const std::string& seq) const {
    NVTX3_FUNC_RANGE();
    std::vector<size_t> context_hits;
    const auto& motif = m_config.motif;
    const auto motif_offset = m_config.motif_offset;
    size_t kmer_len = motif.size();
    size_t search_pos = 0;
    while (search_pos < seq.size() - kmer_len + 1) {
        search_pos = seq.find(motif, search_pos);
        if (search_pos != std::string::npos) {
            context_hits.push_back(search_pos + motif_offset);
            ++search_pos;
        }
    }
    return context_hits;
}

}  // namespace dorado
