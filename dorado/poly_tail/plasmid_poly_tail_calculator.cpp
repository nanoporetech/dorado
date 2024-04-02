#include "plasmid_poly_tail_calculator.h"

#include "read_pipeline/messages.h"
#include "utils/sequence_utils.h"

#include <edlib.h>
#include <spdlog/spdlog.h>

namespace dorado::poly_tail {

SignalAnchorInfo PlasmidPolyTailCalculator::determine_signal_anchor_and_strand(
        const SimplexRead& read) const {
    const std::string& front_flank = m_config.plasmid_front_flank;
    const std::string& rear_flank = m_config.plasmid_rear_flank;
    const std::string& front_flank_rc = m_config.rc_plasmid_front_flank;
    const std::string& rear_flank_rc = m_config.rc_plasmid_rear_flank;
    const int threshold = m_config.flank_threshold;

    std::string_view seq_view = std::string_view(read.read_common.seq);
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_LOC;
    align_config.mode = EDLIB_MODE_HW;

    // Check for forward strand.
    EdlibAlignResult fwd_v1 = edlibAlign(front_flank.data(), int(front_flank.length()),
                                         seq_view.data(), int(seq_view.length()), align_config);
    EdlibAlignResult fwd_v2 = edlibAlign(rear_flank.data(), int(rear_flank.length()),
                                         seq_view.data(), int(seq_view.length()), align_config);

    // Check for reverse strand.
    EdlibAlignResult rev_v1 = edlibAlign(rear_flank_rc.data(), int(rear_flank_rc.length()),
                                         seq_view.data(), int(seq_view.length()), align_config);
    EdlibAlignResult rev_v2 = edlibAlign(front_flank_rc.data(), int(front_flank_rc.length()),
                                         seq_view.data(), int(seq_view.length()), align_config);

    auto scores = {fwd_v1.editDistance, fwd_v2.editDistance, rev_v1.editDistance,
                   rev_v2.editDistance};

    if (std::none_of(std::begin(scores), std::end(scores),
                     [threshold](auto val) { return val < threshold; })) {
        spdlog::trace("{} flank edit distance too high {}", read.read_common.read_id,
                      *std::min_element(std::begin(scores), std::end(scores)));
        return {false, -1, 0, false};
    }

    bool fwd = std::distance(std::begin(scores),
                             std::min_element(std::begin(scores), std::end(scores))) < 2;

    EdlibAlignResult& front_result = fwd ? fwd_v1 : rev_v1;
    EdlibAlignResult& rear_result = fwd ? fwd_v2 : rev_v2;

    // good flank detection with the front and rear in order is the only configuration
    // where we can be sure we haven't cleaved the tail
    bool whole_tail = front_result.editDistance < threshold &&
                      rear_result.editDistance < threshold &&
                      front_result.endLocations[0] < rear_result.startLocations[0];

    int base_anchor = front_result.endLocations[0];
    if (front_result.editDistance - rear_result.editDistance > threshold) {
        // front sequence cleaved?
        base_anchor = rear_result.startLocations[0];
    }

    size_t trailing_tail_bases = 0;
    if (fwd) {
        if (fwd_v1.editDistance < threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(front_flank, 'A');
        }
        if (fwd_v2.editDistance < threshold) {
            trailing_tail_bases += dorado::utils::count_leading_chars(rear_flank, 'A');
        }
    } else {
        if (rev_v1.editDistance < threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(rear_flank_rc, 'T');
        }
        if (rev_v2.editDistance < threshold) {
            trailing_tail_bases += dorado::utils::count_leading_chars(front_flank_rc, 'T');
        }
    }

    edlibFreeAlignResult(fwd_v1);
    edlibFreeAlignResult(fwd_v2);
    edlibFreeAlignResult(rev_v1);
    edlibFreeAlignResult(rev_v2);

    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map = dorado::utils::moves_to_map(read.read_common.moves, stride,
                                                            read.read_common.get_raw_data_samples(),
                                                            read.read_common.seq.size() + 1);
    int signal_anchor = int(seq_to_sig_map[base_anchor]);

    return {fwd, signal_anchor, static_cast<int>(trailing_tail_bases), !whole_tail};
}

}  // namespace dorado::poly_tail
