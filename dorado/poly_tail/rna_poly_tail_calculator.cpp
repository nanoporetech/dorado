#include "rna_poly_tail_calculator.h"

#include "read_pipeline/messages.h"
#include "utils/math_utils.h"

#include <algorithm>

namespace dorado::poly_tail {

float RNAPolyTailCalculator::average_samples_per_base(const std::vector<float>& sizes) const {
    auto quantiles = dorado::utils::quantiles(sizes, {0.1f, 0.9f});
    float sum = 0.f;
    int count = 0;
    for (auto s : sizes) {
        if (s >= quantiles[0] && s <= quantiles[1]) {
            sum += s;
            count++;
        }
    }
    return (count > 0 ? (sum / count) : 0.f);
}

SignalAnchorInfo RNAPolyTailCalculator::determine_signal_anchor_and_strand(
        const SimplexRead& read) const {
    return SignalAnchorInfo{false, read.read_common.rna_adapter_end_signal_pos, 0, false};
}

// Create an offset for dRNA data. There is a tendency to overestimate the length of dRNA
// tails, especially shorter ones. This correction factor appears to fix the bias
// for most dRNA data. This exponential fit was done based on the standards data.
// TODO: In order to improve this, perhaps another pass over the tail interval is needed
// to get a more refined boundary estimation?
int RNAPolyTailCalculator::signal_length_adjustment(int signal_len) const {
    return int(std::round(
            std::min(100.f, std::exp(5.6838f - 0.0021f * static_cast<float>(signal_len)))));
}

std::pair<int, int> RNAPolyTailCalculator::signal_range(int signal_anchor,
                                                        int signal_len,
                                                        float samples_per_base) const {
    const int kSpread = int(std::round(samples_per_base * max_tail_length()));
    return {std::max(0, signal_anchor - 50), std::min(signal_len, signal_anchor + kSpread)};
}

}  // namespace dorado::poly_tail
