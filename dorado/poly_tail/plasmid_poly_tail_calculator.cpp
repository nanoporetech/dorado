#include "plasmid_poly_tail_calculator.h"

#include "read_pipeline/messages.h"
#include "utils/PostCondition.h"
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
    const float threshold = m_config.flank_threshold;

    std::string_view seq_view = std::string_view(read.read_common.seq);
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_LOC;
    align_config.mode = EDLIB_MODE_HW;

    // Check for forward strand.
    EdlibAlignResult fwd_front = edlibAlign(front_flank.data(), int(front_flank.length()),
                                            seq_view.data(), int(seq_view.length()), align_config);
    EdlibAlignResult fwd_rear = edlibAlign(rear_flank.data(), int(rear_flank.length()),
                                           seq_view.data(), int(seq_view.length()), align_config);

    // Check for reverse strand.
    EdlibAlignResult rev_front = edlibAlign(rear_flank_rc.data(), int(rear_flank_rc.length()),
                                            seq_view.data(), int(seq_view.length()), align_config);
    EdlibAlignResult rev_rear = edlibAlign(front_flank_rc.data(), int(front_flank_rc.length()),
                                           seq_view.data(), int(seq_view.length()), align_config);

    auto clear_edlib_results =
            utils::PostCondition([&fwd_front, &fwd_rear, &rev_front, &rev_rear]() {
                edlibFreeAlignResult(fwd_front);
                edlibFreeAlignResult(fwd_rear);
                edlibFreeAlignResult(rev_front);
                edlibFreeAlignResult(rev_rear);
            });

    float fwd_front_score = 1.f - fwd_front.editDistance / float(front_flank.length());
    float fwd_rear_score = 1.f - fwd_rear.editDistance / float(rear_flank.length());
    float rev_front_score = 1.f - rev_front.editDistance / float(rear_flank_rc.length());
    float rev_rear_score = 1.f - rev_rear.editDistance / float(front_flank_rc.length());

    spdlog::trace("Flank scores: fwd_front {} fwd_rear {}, rev_front {}, rev_rear {}",
                  fwd_front_score, fwd_rear_score, rev_front_score, rev_rear_score);

    auto scores = {fwd_front_score, fwd_rear_score, rev_front_score, rev_rear_score};

    if (std::all_of(std::begin(scores), std::end(scores),
                    [threshold](auto val) { return val < threshold; })) {
        spdlog::trace("{} flank scores too low - best score {}", read.read_common.read_id,
                      *std::max_element(std::begin(scores), std::end(scores)));
        return {false, -1, 0, false};
    }

    bool fwd = std::distance(std::begin(scores),
                             std::max_element(std::begin(scores), std::end(scores))) < 2;

    float front_result_score = fwd ? fwd_front_score : rev_front_score;
    float rear_result_score = fwd ? fwd_rear_score : rev_rear_score;
    EdlibAlignResult& front_result = fwd ? fwd_front : rev_front;
    EdlibAlignResult& rear_result = fwd ? fwd_rear : rev_rear;

    // good flank detection with the front and rear in order is the only configuration
    // where we can be sure we haven't cleaved the tail
    bool split_tail = front_result_score >= threshold && rear_result_score >= threshold &&
                      rear_result.endLocations[0] < front_result.startLocations[0];

    if (split_tail) {
        spdlog::trace("{} split tail found - not supported yet", read.read_common.read_id);
        return {false, -1, 0, false};
    }

    int base_anchor = -1;
    size_t trailing_tail_bases = 0;
    if (fwd) {
        if (fwd_front_score < fwd_rear_score) {
            base_anchor = front_result.endLocations[0];
            spdlog::trace("Using fwd front flank as anchor");
        } else {
            base_anchor = rear_result.startLocations[0];
            spdlog::trace("Using fwd rear flank as anchor");
        }

        if (fwd_front_score >= threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(front_flank, 'A');
        }
        if (fwd_rear_score >= threshold) {
            trailing_tail_bases += dorado::utils::count_leading_chars(rear_flank, 'A');
        }
    } else {
        if (rev_front_score < rev_rear_score) {
            base_anchor = front_result.endLocations[0];
            spdlog::trace("Using rev front flank as anchor");
        } else {
            base_anchor = rear_result.startLocations[0];
            spdlog::trace("Using rev rear flank as anchor");
        }

        if (rev_front_score >= threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(rear_flank_rc, 'T');
        }
        if (rev_rear_score >= threshold) {
            trailing_tail_bases += dorado::utils::count_leading_chars(front_flank_rc, 'T');
        }
    }

    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map = dorado::utils::moves_to_map(read.read_common.moves, stride,
                                                            read.read_common.get_raw_data_samples(),
                                                            read.read_common.seq.size() + 1);
    int signal_anchor = int(seq_to_sig_map[base_anchor]);

    return {fwd, signal_anchor, static_cast<int>(trailing_tail_bases), split_tail};
}

std::pair<int, int> PlasmidPolyTailCalculator::signal_range(int signal_anchor,
                                                            int signal_len,
                                                            float samples_per_base,
                                                            [[maybe_unused]] bool fwd) const {
    // We don't know if we found the front or rear flank as the anchor,
    // so search the signal space in both directions
    const int kSpread = int(std::round(samples_per_base * max_tail_length()));
    return {std::max(0, static_cast<int>(signal_anchor - kSpread)),
            std::min(signal_len, static_cast<int>(signal_anchor + kSpread))};
}

}  // namespace dorado::poly_tail
