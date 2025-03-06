#include "MotifMatcher.h"

#include "config/ModBaseModelConfig.h"

#include <nvtx3/nvtx3.hpp>

#include <iterator>
#include <regex>
#include <unordered_map>

namespace {
const std::unordered_map<char, std::string> IUPAC_CODES =
        {
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
    std::string motif_regex = "(";
    for (auto base : motif) {
        motif_regex += IUPAC_CODES.at(base);
    }
    motif_regex += ")";
    return motif_regex;
}

}  // namespace

namespace dorado::modbase {

MotifMatcher::MotifMatcher(const config::ModBaseModelConfig& model_config)
        : MotifMatcher(model_config.mods.motif, model_config.mods.motif_offset) {}

MotifMatcher::MotifMatcher(const std::string& motif, size_t offset)
        : m_motif(expand_motif_regex(motif)), m_motif_offset(offset) {}

std::vector<size_t> MotifMatcher::get_motif_hits(std::string_view seq) const {
    NVTX3_FUNC_RANGE();
    std::vector<size_t> context_hits;

    std::regex regex(m_motif);
    auto start = std::cbegin(seq);
    auto end = std::cend(seq);
    auto pos = start;
    // string_view on linux uses `const_iterator=const char*`,
    // but on Windows it's a `std::_String_view_iterator<_Traits>`
    std::match_results<decltype(pos)> motif_match;
    while (std::regex_search(pos, end, motif_match, regex)) {
        auto hit = std::distance(start, pos) + motif_match.position(0) + m_motif_offset;
        context_hits.push_back(hit);
        pos += motif_match.position(0) + 1;
    }
    return context_hits;
}

}  // namespace dorado::modbase
