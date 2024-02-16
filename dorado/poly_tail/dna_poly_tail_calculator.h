#pragma once

#include "poly_tail_calculator.h"

namespace dorado::poly_tail {

class DNAPolyTailCalculator : public PolyTailCalculator {
public:
    DNAPolyTailCalculator(PolyTailConfig config) : PolyTailCalculator(std::move(config)) {}
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(int) const override { return 0; };
    float min_avg_val() const override { return -3.0f; }
    std::pair<int, int> signal_range(int signal_anchor,
                                     int signal_len,
                                     float samples_per_base) const override;
};

}  // namespace dorado::poly_tail
