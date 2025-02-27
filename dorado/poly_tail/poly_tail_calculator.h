#pragma once

#include "poly_tail_config.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {
class SimplexRead;
}

namespace dorado::poly_tail {

struct SignalAnchorInfo {
    // Is the strand in forward or reverse direction.
    bool is_fwd_strand = true;
    // The start or end anchor for the polyA/T signal
    // depending on whether the strand is forward or
    // reverse.
    int signal_anchor = -1;
    // Number of additional A/T bases in the polyA
    // stretch from the adapter.
    int trailing_adapter_bases = 0;
    // Whether the polyA/T tail is split between the front/end of the read
    // This can only be true for plasmids
    bool split_tail = false;
};

class PolyTailCalculator {
public:
    PolyTailCalculator(PolyTailConfig config, float speed_calibration, float offset_calibration)
            : m_config(std::move(config)),
              m_speed_calibration(speed_calibration),
              m_offset_calibration(offset_calibration) {}

    virtual ~PolyTailCalculator() = default;

    // returns information about the polyA/T tail. signal_anchor = -1 on failure
    virtual SignalAnchorInfo determine_signal_anchor_and_strand(const SimplexRead& read) const = 0;

    // returns the number of bases in the polyA/T tail, or -1 on failure
    int calculate_num_bases(const SimplexRead& read, const SignalAnchorInfo& signal_info) const;

    static int max_tail_length() { return 750; };

protected:
    // calculate the average number of samples per base
    virtual float average_samples_per_base(const std::vector<float>& sizes) const = 0;

    // Returns any adjustment required for the provided signal_len
    virtual int signal_length_adjustment(const SimplexRead& read, int signal_len) const = 0;

    // Floor for average signal value of poly tail.
    virtual float min_avg_val() const = 0;

    // Returns the acceptable distance between the supplied interval and the anchor
    virtual std::pair<int, int> buffer_range(const std::pair<int, int>& interval,
                                             float samples_per_base) const = 0;

    // Determine the outer boundary of the signal space to consider based on the anchor.
    virtual std::pair<int, int> signal_range(int signal_anchor,
                                             int signal_len,
                                             float samples_per_base,
                                             bool fwd) const;

    std::pair<float, float> estimate_samples_per_base(const dorado::SimplexRead& read) const;

    // Find the signal range near the provided anchor that corresponds to the polyA/T tail
    std::pair<int, int> determine_signal_bounds(int signal_anchor,
                                                bool fwd,
                                                const SimplexRead& read,
                                                float num_samples_per_base,
                                                float std_samples_per_base) const;

    const PolyTailConfig m_config;
    const float m_speed_calibration;
    const float m_offset_calibration;
};

class PolyTailCalculatorFactory {
public:
    static std::shared_ptr<const PolyTailCalculator> create(const PolyTailConfig& config,
                                                            bool is_rna,
                                                            bool is_rna_adapter,
                                                            float speed_calibration,
                                                            float offset_calibration);
};

}  // namespace dorado::poly_tail
