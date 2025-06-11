#pragma once

#include "poly_tail_config.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dorado {
class SimplexRead;
}

namespace dorado::poly_tail {

enum class SearchDirection {
    BACKWARD = 1,
    FORWARD = 2,
};

struct SignalAnchorInfo {
    // Search direction from the anchor sample in signal space
    SearchDirection search_dir = SearchDirection::BACKWARD;
    // The anchor sample for the polyA/T signal search
    int signal_anchor = -1;
    // Number of additional A/T bases in the polyA
    // stretch from the adapter.
    int trailing_adapter_bases = 0;
};

struct PolyTailLengthInfo {
    // the length of polyA/T tail (or -1 on failure)
    int num_bases = -1;
    // the range of the polyA/T tail in the raw signal
    std::pair<int, int> signal_range = {-1, -1};
    std::pair<int, int> split_signal_range = {-1, -1};
};

struct PolyTailCalibrationCoeffs {
    std::optional<float> speed;
    std::optional<float> offset;
};

class PolyTailCalculator {
public:
    PolyTailCalculator(PolyTailConfig config, const PolyTailCalibrationCoeffs& calibration)
            : m_config(std::move(config)), m_calibration(calibration) {}

    virtual ~PolyTailCalculator() = default;

    // returns information about the polyA/T tail. signal_anchor = -1 on failure
    virtual std::vector<SignalAnchorInfo> determine_signal_anchor_and_strand(
            const SimplexRead& read) const = 0;

    // returns a struct with: number of bases in the polyA/T tail (), start and end of poly(A) in raw signal (all -1 on failure)
    PolyTailLengthInfo calculate_num_bases(const SimplexRead& read,
                                           const std::vector<SignalAnchorInfo>& signal_info) const;

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
                                             SearchDirection direction) const;

    std::pair<float, float> estimate_samples_per_base(const dorado::SimplexRead& read) const;

    // Find the signal range near the provided anchor that corresponds to the polyA/T tail
    std::pair<int, int> determine_signal_bounds(int signal_anchor,
                                                SearchDirection direction,
                                                const SimplexRead& read,
                                                float num_samples_per_base,
                                                float std_samples_per_base) const;

    const PolyTailConfig m_config;
    const PolyTailCalibrationCoeffs m_calibration;
};

class PolyTailCalculatorFactory {
public:
    static std::shared_ptr<const PolyTailCalculator> create(
            const PolyTailConfig& config,
            bool is_rna,
            bool is_rna_adapter,
            const PolyTailCalibrationCoeffs& calibration);
};

}  // namespace dorado::poly_tail
