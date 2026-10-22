#pragma once

#include "poly_tail/poly_tail_calculator.h"

namespace dorado::poly_tail {

class RNAPolyTailCalculator : public PolyTailCalculator {
public:
    RNAPolyTailCalculator(PolyTailConfig config, const PolyTailCalibrationCoeffs& calibration);
    std::vector<SignalAnchorInfo> determine_signal_anchor_and_strand(
            const SimplexRead& read,
            bool has_rna_adapters) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(const SimplexRead& read, int signal_len) const override;
    float min_avg_val() const override { return -0.5f; }
    std::pair<int, int> buffer_range(const std::pair<int, int>& interval,
                                     bool has_rna_adapters,
                                     float num_samples_per_base) const override;
    std::pair<int, int> signal_range(int signal_anchor,
                                     int signal_len,
                                     float samples_per_base,
                                     SearchDirection direction) const override;
};

}  // namespace dorado::poly_tail
