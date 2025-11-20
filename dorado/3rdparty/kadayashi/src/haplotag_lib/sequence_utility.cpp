#include "sequence_utility.h"

#include <algorithm>
#include <cstdint>

namespace kadayashi {

void seq2nt4seq(const char *seq, const int seq_l, std::vector<uint8_t> &h) {
    h.resize(seq_l);
    for (int i = 0; i < seq_l; i++) {
        h[i] = static_cast<uint8_t>(kadayashi::kdy_seq_nt4_table[static_cast<int>(seq[i])]);
    }
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

int diff_of_top_two(std::vector<uint32_t> &d) {
    std::stable_sort(d.begin(), d.end());
    return d[3] - d[2];
}

uint32_t max_of_u32_vec(const std::vector<uint32_t> &d, int *idx) {
    const auto rit = std::max_element(d.rbegin(), d.rend());
    if (idx) {
        *idx = std::distance(rit, d.rend()) - 1;
    }
    return *rit;
}

}  // namespace kadayashi
