#include "plasmid_poly_tail_calculator.h"

#include "read_pipeline/base/messages.h"
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

std::vector<SignalAnchorInfo> PlasmidPolyTailCalculator::determine_signal_anchor_and_strand(
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
    const Result& front_result = fwd ? fwd_front : rev_front;
    const Result& rear_result = fwd ? fwd_rear : rev_rear;

    // front and rear good but out of order indicates we've cleaved the tail
    bool split_tail = (front_result_score >= threshold) && (rear_result_score >= threshold) &&
                      (rear_result.end < front_result.start);

    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map = dorado::utils::moves_to_map(read.read_common.moves, stride,
                                                            read.read_common.get_raw_data_samples(),
                                                            read.read_common.seq.size() + 1);

    std::vector<SignalAnchorInfo> signal_info;
    if (fwd) {
        if (fwd_front.score >= threshold) {
            int trailing_tail_bases =
                    static_cast<int>(dorado::utils::count_trailing_chars(front_flank, 'A'));
            int anchor = int(seq_to_sig_map[front_result.end]);
            signal_info.emplace_back(
                    SignalAnchorInfo{SearchDirection::FORWARD, anchor, trailing_tail_bases});
        }

        if ((split_tail || std::empty(signal_info)) && fwd_rear.score >= threshold) {
            int trailing_tail_bases =
                    static_cast<int>(dorado::utils::count_leading_chars(rear_flank, 'A'));
            int anchor = int(seq_to_sig_map[rear_result.start]);
            signal_info.emplace_back(
                    SignalAnchorInfo{SearchDirection::BACKWARD, anchor, trailing_tail_bases});
        }
    } else {
        if (rev_front.score >= threshold) {
            int trailing_tail_bases =
                    static_cast<int>(dorado::utils::count_trailing_chars(rear_flank_rc, 'T'));
            int anchor = int(seq_to_sig_map[front_result.end]);
            signal_info.emplace_back(
                    SignalAnchorInfo{SearchDirection::FORWARD, anchor, trailing_tail_bases});
        }

        if ((split_tail || std::empty(signal_info)) && rev_rear.score >= threshold) {
            int trailing_tail_bases =
                    static_cast<int>(dorado::utils::count_leading_chars(front_flank_rc, 'T'));
            int anchor = int(seq_to_sig_map[rear_result.start]);
            signal_info.emplace_back(
                    SignalAnchorInfo{SearchDirection::BACKWARD, anchor, trailing_tail_bases});
        }
    }

    return signal_info;
}

}  // namespace dorado::poly_tail
