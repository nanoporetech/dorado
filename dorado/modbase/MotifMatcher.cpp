#include "MotifMatcher.h"

#include "nn/ModBaseModelConfig.h"

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

std::string expand_motif_regex(const std::string& motif) {
    std::ostringstream motif_regex_ss;
    motif_regex_ss << "(";
    for (auto base : motif) {
        motif_regex_ss << IUPAC_CODES.at(base);
    }
    motif_regex_ss << ")";
    return motif_regex_ss.str();
}

}  // namespace

namespace dorado {

MotifMatcher::MotifMatcher(const ModBaseModelConfig& model_config)
        : MotifMatcher(model_config.motif, model_config.motif_offset) {}

MotifMatcher::MotifMatcher(const std::string& motif, size_t offset)
        : m_motif(expand_motif_regex(motif)), m_motif_offset(offset) {}

std::vector<size_t> MotifMatcher::get_motif_hits(const std::string& seq) const {
    NVTX3_FUNC_RANGE();
    std::vector<size_t> context_hits;

    std::regex regex(m_motif);
    std::smatch motif_match;
    auto start = std::cbegin(seq);
    auto end = std::cend(seq);
    auto pos = start;
    while (std::regex_search(pos, end, motif_match, regex)) {
        auto hit = std::distance(start, pos) + motif_match.position(0) + m_motif_offset;
        context_hits.push_back(hit);
        pos += motif_match.position(0) + 1;
    }
    return context_hits;
}

bool MotifMatcher::matches_motif(std::string_view seq) const {
    std::regex regex(m_motif);
    return std::regex_match(seq.begin(), seq.end(), regex);
}

}  // namespace dorado
