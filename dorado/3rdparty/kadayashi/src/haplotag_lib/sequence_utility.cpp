#include "sequence_utility.h"

#include <algorithm>
#include <cstdint>

namespace kadayashi {

std::vector<uint8_t> seq2nt4seq(const std::string_view &seq) {
    std::vector<uint8_t> h;
    h.reserve(seq.size());
    for (size_t i = 0; i < seq.size(); i++) {
        h.emplace_back(
                static_cast<uint8_t>(kadayashi::kdy_seq_nt4_table[static_cast<int>(seq[i])]));
    }
    return h;
}

std::string nt4seq2seq(const std::vector<uint8_t> &allele_vec) {
    const int l =
            static_cast<int>(allele_vec.size()) -
            1;  // allele seq should always be at least length 2 (1base + 1cigar op by the end)
    std::string ret(l, '\0');
    for (int i = 0; i < l; i++) {
        ret[i] = "ACGTNN"[allele_vec[i]];
    }
    return ret;
}

bool max_of_u32_arr(const std::span<const uint32_t> &d, int *idx, uint32_t *val) {
    if (d.empty()) {
        return false;
    } else {
        const auto rit = std::max_element(d.begin(), d.end());
        if (idx) {
            *idx = static_cast<int>(std::distance(d.begin(), rit));
        }
        if (val) {
            *val = *rit;
        }
    }
    return true;
}

}  // namespace kadayashi
