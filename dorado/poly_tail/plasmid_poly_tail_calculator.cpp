#include "plasmid_poly_tail_calculator.h"

#include "read_pipeline/messages.h"
#include "utils/PostCondition.h"
#include "utils/log_utils.h"
#include "utils/sequence_utils.h"

#include <edlib.h>
#include <spdlog/spdlog.h>

namespace dorado::poly_tail {

namespace {
struct Result {
    float score{-1.f};
    int start{-1};
    int end{-1};
};

}  // namespace

SignalAnchorInfo PlasmidPolyTailCalculator::determine_signal_anchor_and_strand(
        const SimplexRead& read) const {
    const std::string_view front_flank = m_config.plasmid_front_flank;
    const std::string_view rear_flank = m_config.plasmid_rear_flank;
    const std::string_view front_flank_rc = m_config.rc_plasmid_front_flank;
    const std::string_view rear_flank_rc = m_config.rc_plasmid_rear_flank;
    const float threshold = m_config.flank_threshold;

    std::string_view seq_view{read.read_common.seq};
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_LOC;
    align_config.mode = EDLIB_MODE_HW;

    auto align_query = [&threshold, &align_config, &seq_view](
                               const std::string_view& query,
                               const std::string_view& desc) -> Result {
        if (std::empty(query)) {
            return {};
        }

        EdlibAlignResult align_result =
                edlibAlign(query.data(), int(query.length()), seq_view.data(),
                           int(seq_view.length()), align_config);
        Result result;
        result.score = 1.f - align_result.editDistance / float(query.length());
        dorado::utils::trace_log("Flank score: {} {}", desc, result.score);
        if (result.score >= threshold) {
            result.start = align_result.startLocations[0];
            result.end = align_result.endLocations[0];
        }
        edlibFreeAlignResult(align_result);
        return result;
    };

    // Check for forward strand.
    auto fwd_front = align_query(front_flank, "FWD_FRONT");
    auto fwd_rear = align_query(rear_flank, "FWD_REAR");

    // Check for reverse strand.
    auto rev_front = align_query(rear_flank_rc, "REV_FRONT");
    auto rev_rear = align_query(front_flank_rc, "REV_REAR");

    auto scores = {fwd_front.score, fwd_rear.score, rev_front.score, rev_rear.score};
    bool fwd = std::distance(std::begin(scores),
                             std::max_element(std::begin(scores), std::end(scores))) <
               static_cast<int>(std::size(scores) / 2);

    float front_result_score = fwd ? fwd_front.score : rev_front.score;
    float rear_result_score = fwd ? fwd_rear.score : rev_rear.score;
    Result& front_result = fwd ? fwd_front : rev_front;
    Result& rear_result = fwd ? fwd_rear : rev_rear;

    // front and rear good but out of order indicates we've cleaved the tail
    bool split_tail = (front_result_score >= threshold) && (rear_result_score >= threshold) &&
                      (rear_result.end < front_result.start);
    std::array<int, 2> anchors{-1, -1};
    size_t trailing_tail_bases = 0;
    int anchor_index = 0;
    if (fwd) {
        if (fwd_front.score >= threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(front_flank, 'A');
            anchors[anchor_index] = front_result.end;
            ++anchor_index;
        }

        if ((split_tail || anchor_index == 0) && fwd_rear.score >= threshold) {
            anchors[anchor_index] = rear_result.start;
            trailing_tail_bases += dorado::utils::count_leading_chars(rear_flank, 'A');
            ++anchor_index;
        }
    } else {
        if (rev_front.score >= threshold) {
            trailing_tail_bases += dorado::utils::count_trailing_chars(rear_flank_rc, 'T');
            anchors[anchor_index] = front_result.end;
            ++anchor_index;
        }

        if ((split_tail || anchor_index == 0) && rev_rear.score >= threshold) {
            anchors[anchor_index] = rear_result.start;
            trailing_tail_bases += dorado::utils::count_leading_chars(front_flank_rc, 'T');
            ++anchor_index;
        }
    }

    assert(anchor_index <= static_cast<int>(std::size(anchors)));

    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map = dorado::utils::moves_to_map(read.read_common.moves, stride,
                                                            read.read_common.get_raw_data_samples(),
                                                            read.read_common.seq.size() + 1);
    for (int& anchor : anchors) {
        if (anchor != -1) {
            anchor = int(seq_to_sig_map[anchor]);
        }
    }

    return {fwd, anchors[0], static_cast<int>(trailing_tail_bases), anchors[1]};
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
