#include "ModBaseContext.h"

#include "MotifMatcher.h"
#include "utils/sequence_utils.h"

#include <sstream>

namespace dorado::modbase {

ModBaseContext::ModBaseContext() {}
ModBaseContext::~ModBaseContext() {}

const std::string& ModBaseContext::motif(char base) const {
    return m_motifs[utils::base_to_int(base)];
}

size_t ModBaseContext::motif_offset(char base) const { return m_offsets[utils::base_to_int(base)]; }

void ModBaseContext::set_context(std::string motif, size_t offset) {
    if (motif.size() < 2) {
        // empty motif, or just the canonical base
        return;
    }
    char base = motif.at(offset);
    auto index = utils::base_to_int(base);
    m_motif_matchers[index] = std::make_unique<MotifMatcher>(motif, offset);
    m_motifs[index] = std::move(motif);
    m_offsets[index] = offset;
}

bool ModBaseContext::decode(const std::string& context_string) {
    std::vector<std::string> tokens;
    std::istringstream context_stream(context_string);
    std::string token;
    while (std::getline(context_stream, token, ':')) {
        tokens.push_back(token);
    }
    if (tokens.size() != 4) {
        return false;
    }
    auto canonical = "ACGT";
    for (size_t i = 0; i < 4; ++i) {
        if (tokens[i] == "_") {
            m_motif_matchers[i].reset();
            m_motifs[i].clear();
            m_offsets[i] = 0;
        } else {
            auto x = tokens[i].find('X');
            if (x == std::string::npos) {
                return false;
            }
            m_motifs[i] = tokens[i];
            m_motifs[i][x] = canonical[i];
            m_offsets[i] = x;
            m_motif_matchers[i] = std::make_unique<MotifMatcher>(m_motifs[i], m_offsets[i]);
        }
    }
    return true;
}

std::string ModBaseContext::encode() const {
    std::ostringstream s;
    for (size_t i = 0; i < 4; ++i) {
        if (m_motifs[i].empty()) {
            s << '_';
        } else {
            auto m = m_motifs[i];
            m[m_offsets[i]] = 'X';
            s << m;
        }
        if (i < 3) {
            s << ':';
        }
    }
    return s.str();
}

std::vector<bool> ModBaseContext::get_sequence_mask(std::string_view sequence) const {
    std::vector<bool> mask(sequence.size(), false);
    for (auto& matcher : m_motif_matchers) {
        if (matcher) {
            auto hits = matcher->get_motif_hits(sequence);
            for (auto hit : hits) {
                mask[hit] = true;
            }
        }
    }
    return mask;
}

void ModBaseContext::update_mask(std::vector<bool>& mask,
                                 const std::string& sequence,
                                 const std::vector<std::string>& modbase_alphabet,
                                 const std::vector<uint8_t>& modbase_probs,
                                 uint8_t threshold) const {
    // Iterate over the provided alphabet and find all the bases that may be modified.
    size_t num_channels = modbase_alphabet.size();
    const std::string cardinal_bases = "ACGT";
    char current_cardinal = 0;
    for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        if (cardinal_bases.find(modbase_alphabet[channel_idx]) != std::string::npos) {
            // A cardinal base.
            current_cardinal = modbase_alphabet[channel_idx][0];
        } else {
            if (!m_motifs[utils::base_to_int(current_cardinal)].empty()) {
                // This cardinal base has a context associated with modifications, so the mask should
                // not be updated, regardless of the threshold.
                continue;
            }
            for (size_t base_idx = 0; base_idx < sequence.size(); base_idx++) {
                if (sequence[base_idx] == current_cardinal) {
                    if (modbase_probs[base_idx * num_channels + channel_idx] >= threshold) {
                        mask[base_idx] = true;
                    }
                }
            }
        }
    }
}

}  // namespace dorado::modbase