#pragma once

#include "poly_tail_calculator.h"

namespace dorado::poly_tail {

class RNAPolyTailCalculator : public PolyTailCalculator {
public:
    RNAPolyTailCalculator(PolyTailConfig config,
                          bool is_rna_adapter,
                          float speed_calibration,
                          float offset_calibration);
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(const SimplexRead& read, int signal_len) const override;
    float min_avg_val() const override { return -0.5f; }
    std::pair<int, int> buffer_range(const std::pair<int, int>& interval,
                                     float samples_per_base) const override;
    std::pair<int, int> signal_range(int signal_anchor,
                                     int signal_len,
                                     float samples_per_base,
                                     bool fwd) const override;

private:
    bool m_rna_adapter;
};

}  // namespace dorado::poly_tail
