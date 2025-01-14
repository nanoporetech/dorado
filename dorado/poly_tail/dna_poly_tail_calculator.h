#pragma once

#include "poly_tail_calculator.h"

namespace dorado::poly_tail {

class DNAPolyTailCalculator : public PolyTailCalculator {
public:
    DNAPolyTailCalculator(PolyTailConfig config, float speed_calibration, float offset_calibration)
            : PolyTailCalculator(std::move(config), speed_calibration, offset_calibration) {}
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;

protected:
    float average_samples_per_base(const std::vector<float>& sizes) const override;
    int signal_length_adjustment(const SimplexRead& read, int signal_len) const override;
    float min_avg_val() const override { return -3.0f; }
    std::pair<int, int> buffer_range(const std::pair<int, int>& interval,
                                     [[maybe_unused]] float samples_per_base) const override {
        // The buffer is currently the length of the interval
        // itself. This heuristic generally works because a longer interval
        // detected is likely to be the correct one so we relax the
        // how close it needs to be to the anchor to account for errors
        // in anchor determination.
        return {interval.second - interval.first, interval.second - interval.first};
    }
};

}  // namespace dorado::poly_tail
