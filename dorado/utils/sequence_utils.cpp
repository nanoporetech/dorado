#include "sequence_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace utils {

float mean_qscore_from_qstring(const std::string& qstring) {
    if (qstring.empty()) {
        return 0;
    }

    std::vector<float> scores;
    scores.reserve(qstring.length());
    std::transform(qstring.begin(), qstring.end(), std::back_inserter(scores),
                   [](const char& qchar) {
                       float qscore = static_cast<float>(qchar - 33);
                       return std::pow(10.f, -qscore / 10.f);
                   });
    float mean_error = std::accumulate(scores.begin(), scores.end(), 0.f) / scores.size();
    float mean_qscore = -10.0f * log10(mean_error);
    mean_qscore = std::min(90.0f, std::max(1.0f, mean_qscore));
    return mean_qscore;
}

int base_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

std::vector<int> sequence_to_ints(const std::string& sequence) {
    std::vector<int> sequence_ints;
    sequence_ints.reserve(sequence.size());
    std::transform(std::begin(sequence), std::end(sequence), std::back_inserter(sequence_ints),
                   &base_to_int);
    return sequence_ints;
}

std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                   size_t block_stride,
                                   size_t signal_len) {
    std::vector<uint64_t> seq_to_sig_map;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == 1) {
            seq_to_sig_map.push_back(i * block_stride);
        }
    }
    seq_to_sig_map.push_back(signal_len);
    return seq_to_sig_map;
}

}  // namespace utils
