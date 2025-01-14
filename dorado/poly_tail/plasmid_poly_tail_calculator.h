#pragma once

#include "dna_poly_tail_calculator.h"

namespace dorado::poly_tail {

class PlasmidPolyTailCalculator : public DNAPolyTailCalculator {
public:
    PlasmidPolyTailCalculator(PolyTailConfig config,
                              float speed_calibration,
                              float offset_calibration)
            : DNAPolyTailCalculator(std::move(config), speed_calibration, offset_calibration) {}
    SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const override;

protected:
    std::pair<int, int> signal_range(int signal_anchor,
                                     int signal_len,
                                     float samples_per_base,
                                     bool fwd) const override;
};

}  // namespace dorado::poly_tail
