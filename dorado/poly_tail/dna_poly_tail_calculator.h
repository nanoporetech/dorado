#pragma once

#include "poly_tail/poly_tail_calculator.h"

namespace dorado::poly_tail {

class DNAPolyTailCalculator : public PolyTailCalculator {
public:
    DNAPolyTailCalculator(PolyTailConfig config, const PolyTailCalibrationCoeffs& calibration)
            : PolyTailCalculator(std::move(config), calibration) {}
    std::vector<SignalAnchorInfo> determine_signal_anchor_and_strand(
            const SimplexRead& read,
            bool has_rna_adapters) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(const SimplexRead& read, int signal_len) const override;
    float min_avg_val() const override { return -3.0f; }
};

}  // namespace dorado::poly_tail
