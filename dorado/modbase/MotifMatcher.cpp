#include "MotifMatcher.h"

#include <nvtx3/nvtx3.hpp>

#include <regex>
#include <sstream>
#include <unordered_map>

namespace {
const std::unordered_map<char, std::string> IUPAC_CODES = {
        // clang-format off
        {'A', "A"},
        {'C', "C"},
        {'G', "G"},
        {'T', "T"},
        {'U', "T"},  // basecalls will have "T"s instead of "U"s
        {'R', "[AG]"},
        {'Y', "[CT]"}, 
        {'S', "[GC]"}, 
        {'W', "[AT]"},
        {'K', "[GT]"}, 
        {'M', "[AC]"}, 
        {'B', "[CGT]"},
        {'D', "[AGT]"},
        {'H', "[ACT]"},
        {'V', "[ACG]"},
        {'N', "[ACGT]"},
        // clang-format on
};
}

namespace dorado {

MotifMatcher::MotifMatcher(const ModBaseModelConfig& model_config) : m_config(model_config) {}

std::vector<size_t> MotifMatcher::get_motif_hits(const std::string& seq) const {
    NVTX3_FUNC_RANGE();
    std::vector<size_t> context_hits;
    std::ostringstream motif_regex_ss;
    motif_regex_ss << "(";
    for (auto base : m_config.motif) {
        motif_regex_ss << IUPAC_CODES.at(base);
    }
    motif_regex_ss << ")";
    const auto motif = motif_regex_ss.str();
    const auto motif_offset = m_config.motif_offset;

    std::regex regex(motif);
    std::smatch motif_match;
    auto start = std::cbegin(seq);
    auto end = std::cend(seq);
    while (std::regex_search(start, end, motif_match, regex)) {
        auto hit = std::distance(std::cbegin(seq), start) + motif_match.position(0) + motif_offset;
        context_hits.push_back(hit);
        start += motif_match.position(0) + 1;
    }
    return context_hits;
}

}  // namespace dorado
