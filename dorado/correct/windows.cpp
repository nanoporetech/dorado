#include "windows.h"

#include "conversions.h"
#include "features.h"
#include "read_pipeline/messages.h"
#include "torch_utils/gpu_profiling.h"
#include "types.h"
#include "utils/paf_utils.h"

#include <spdlog/spdlog.h>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

// #define DORADO_CORRECT_DEBUG_WINDOWS
// #define DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
// #define DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID -1

#if defined(DORADO_CORRECT_DEBUG_WINDOWS) || defined(DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2) || \
        defined(DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)
#include <iostream>
#endif

namespace dorado::correction {

namespace {

std::vector<int32_t> calc_max_step_per_pos(const CorrectionAlignments& alignments,
                                           const std::vector<OverlapWindow>& window,
                                           const int32_t window_size,
                                           const int32_t init_value) {
    utils::ScopedProfileRange spr("calc_max_step_per_pos", 1);

    if (std::empty(alignments.overlaps)) {
        return {};
    }

    std::vector<int32_t> ret(window_size, init_value);

    for (const auto& chunk : window) {
        const int32_t aln_id = chunk.overlap_idx;

        const auto& cigar = alignments.cigars[aln_id];
        const int32_t cig_len = static_cast<int32_t>(std::size(cigar));

        int32_t pos = 0;

        // For each position of the target, determine how many columns there are.
        for (int32_t cig_id = chunk.cigar_start_idx; cig_id <= chunk.cigar_end_idx; ++cig_id) {
            if (cig_id == cig_len) {
                break;
            }
            if ((cig_id == chunk.cigar_end_idx) && (chunk.cigar_end_offset == 0)) {
                break;
            }

            const auto& c = cigar[cig_id];

            int32_t len = static_cast<int32_t>(c.len);
            if (chunk.cigar_start_idx == chunk.cigar_end_idx) {
                len = chunk.cigar_end_offset - chunk.cigar_start_offset;
            } else if (cig_id == chunk.cigar_start_idx) {
                len = static_cast<int32_t>(c.len) - chunk.cigar_start_offset;
            } else if (cig_id == chunk.cigar_end_idx) {
                len = chunk.cigar_end_offset;
            } else {
                len = static_cast<int32_t>(c.len);
            }

            if ((pos > window_size) || ((pos == window_size) && (c.op != CigarOpType::I))) {
                throw std::runtime_error("Position out of bounds! aln_id = " +
                                         std::to_string(aln_id) + ", pos = " + std::to_string(pos) +
                                         ", window_size = " + std::to_string(window_size) +
                                         ", CIGAR op = " + cigar_op_to_string(c));
            }

            if ((c.op == CigarOpType::M) || (c.op == CigarOpType::EQ) || (c.op == CigarOpType::X) ||
                c.op == CigarOpType::D) {
                pos += len;

            } else if ((c.op == CigarOpType::I) && (pos > 0)) {
                ret[pos - 1] = std::max(ret[pos - 1], len + init_value);

            } else {
                throw std::runtime_error(
                        "Unsupported CIGAR operation when computing inclusive scan!");
            }
        }
    }

    return ret;
}

int32_t find_target_end_for_window(const std::vector<int32_t>& consumed,
                                   const int32_t win_tstart,
                                   const int32_t win_tend,
                                   const int32_t max_num_columns) {
    utils::ScopedProfileRange spr("find_target_end_for_window", 1);
    int32_t num_cols = 0;
    for (int32_t tpos = win_tstart; tpos < win_tend; ++tpos) {
        const int32_t curr_max_step = consumed[tpos - win_tstart];
        if ((num_cols + curr_max_step) > max_num_columns) {
            return tpos;
        }
        num_cols += curr_max_step;
    }
    return win_tend;
}

}  // namespace

// This function attempts to create windows from the set
// of reads that align to the target sequence being corrected.
// Each window covers a fixed chunk (i.e. window size) on the
// target sequence. Then based on the alignment information
// (i.e. cigar string), the corresponding intervals of each of
// the query reads is determined.
// The logic of this code has been borrowed from the original
// Rust code from the Herro code base -
// https://github.com/lbcb-sci/herro/blob/main/src/windowing.rs#L41
bool extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                     const CorrectionAlignments& alignments,
                     int window_size) {
    utils::ScopedProfileRange spr("extract_windows", 1);

    int num_alignments = (int)alignments.overlaps.size();
    if (num_alignments == 0) {
        return true;
    }
    const int32_t tlen = alignments.overlaps.front().tlen;
    for (int aln_idx = 0; aln_idx < num_alignments; aln_idx++) {
        const auto& overlap = alignments.overlaps[aln_idx];
        const auto& cigar = alignments.cigars[aln_idx];

        // Following the is_target == False logic form the rust code.
        if ((overlap.tend - overlap.tstart < window_size) ||
            (overlap.qend - overlap.qstart < window_size)) {
            continue;
        }

        LOG_TRACE("qlen {} qstart {} qend {} strand {} tlen {} tstart {} tend {}", overlap.qlen,
                  overlap.qstart, overlap.qend, overlap.fwd, overlap.tlen, overlap.tstart,
                  overlap.tend);

        int zeroth_window_thresh = static_cast<int>(0.1f * window_size);
        int nth_window_thresh = overlap.tlen - zeroth_window_thresh;

        LOG_TRACE("zeroth {} nth {}", zeroth_window_thresh, nth_window_thresh);

        // Keep the first and last windows for only 10% of the read length to capture
        // overhangs from query alignments on either end.
        int first_window = (overlap.tstart < zeroth_window_thresh
                                    ? 0
                                    : (overlap.tstart + window_size - 1) / window_size);
        int last_window = (overlap.tend > nth_window_thresh ? ((overlap.tend - 1) / window_size) + 1
                                                            : overlap.tend / window_size);

        if (first_window < 0 || (last_window - 1) >= (int)windows.size()) {
            spdlog::error(
                    "{} zeroth thres {} nth thres {} first win {} last win {} windows size {} "
                    "overlap "
                    "tlen {} overlsp tstart {} overlap tend {} qlen {} qstart {} qend {}",
                    alignments.read_name, zeroth_window_thresh, nth_window_thresh, first_window,
                    last_window, windows.size(), overlap.tlen, overlap.tstart, overlap.tend,
                    overlap.qlen, overlap.qstart, overlap.qend);
            return false;
        }

        int tstart = overlap.tstart;
        int tpos = overlap.tstart;
        int qpos = 0;

        LOG_TRACE("first window {} last window {} tstart {} tpos {}", first_window, last_window,
                  tstart, tpos);

        if (last_window - first_window < 1) {
            continue;
        }

        int t_window_start = -1;
        int q_window_start = -1;
        int cigar_start_idx = -1;
        int cigar_start_offset = -1;

        LOG_TRACE("tpos {} qpos {}", tpos, qpos);

        if ((tpos % window_size == 0) || (tstart < zeroth_window_thresh)) {
            t_window_start = tpos;
            q_window_start = qpos;
            cigar_start_idx = 0;
            cigar_start_offset = 0;
        }

        LOG_TRACE("t_window_start {} q_window_start {} cigar_start_idx {} cigar_start_offset {}",
                  t_window_start, q_window_start, cigar_start_idx, cigar_start_offset);

        for (int cigar_idx = 0; cigar_idx < (int)cigar.size(); cigar_idx++) {
            auto op = cigar[cigar_idx];
            int tnew = tpos;
            int qnew = qpos;
            switch (op.op) {
            case CigarOpType::EQ:
            case CigarOpType::X:
                tnew = tpos + op.len;
                qnew = qpos + op.len;
                LOG_TRACE("{} {}", op.len, "M");
                break;
            case CigarOpType::D:
                tnew = tpos + op.len;
                LOG_TRACE("{} {}", op.len, "D");
                break;
            case CigarOpType::I:
                qpos += op.len;
                LOG_TRACE("{} {}", op.len, "I");
                continue;
            default:
                throw std::runtime_error("Unexpected CigarOpType in extract_windows: " +
                                         std::string(1, convert_cigar_op_to_char(op.op)));
            }

            LOG_TRACE("tpos {} qpos {} tnew {} qnew {}", tpos, qpos, tnew, qnew);

            const int current_w = tpos / window_size;
            const int new_w = tnew / window_size;
            const int diff_w = new_w - current_w;

            if (diff_w == 0) {
                tpos = tnew;
                qpos = qnew;
                continue;
            }

            // If the cigar operations spans multiple windows, break it up into
            // multiple chunks with intervals corresponding to each window.
            for (int i = 1; i < diff_w; i++) {
                const int offset = (current_w + i) * window_size - tpos;

                const int q_start_new = (op.op == CigarOpType::EQ || op.op == CigarOpType::X)
                                                ? qpos + offset
                                                : qpos;

                if (cigar_start_idx >= 0) {
                    const int32_t current_w_start = (t_window_start / window_size) * window_size;
                    const int32_t current_w_end = std::min(current_w_start + window_size, tlen);
                    windows[(current_w + i) - 1].push_back(
                            {aln_idx, current_w_start, current_w_end, t_window_start, tpos + offset,
                             q_window_start, q_start_new, cigar_start_idx, cigar_start_offset,
                             (int)cigar_idx, offset, 0.f, 0});

                    LOG_TRACE(
                            "pushed t_window_start {} q_window_start {} q_start_new {} "
                            "cigar_start_idx {} cigar_start_offseet {} cigar_idx {} offset {}",
                            t_window_start, q_window_start, q_start_new, cigar_start_idx,
                            cigar_start_offset, cigar_idx, offset);

                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::EQ || op.op == CigarOpType::X) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }

                    cigar_start_idx = (int)cigar_idx;
                    cigar_start_offset = offset;
                } else {
                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::EQ || op.op == CigarOpType::X) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }

                    cigar_start_idx = cigar_idx;
                    cigar_start_offset = offset;
                }
            }

            LOG_TRACE("new_w {} window size {} tpos {}", new_w, window_size, tpos);
            const int offset = new_w * window_size - tpos;

            int qend = (op.op == CigarOpType::EQ || op.op == CigarOpType::X) ? qpos + offset : qpos;

            LOG_TRACE("offset {} qend {}", offset, qend);

            int cigar_end_idx = -1;
            int cigar_end_offset = -1;

            if (tnew == new_w * window_size) {
                if (cigar_idx + 1 < (int)cigar.size() &&
                    cigar[cigar_idx + 1].op == CigarOpType::I) {
                    qend += cigar[cigar_idx + 1].len;
                    cigar_end_idx = cigar_idx + 2;
                } else {
                    cigar_end_idx = cigar_idx + 1;
                }

                cigar_end_offset = 0;
            } else {
                cigar_end_idx = cigar_idx;
                cigar_end_offset = offset;
            }

            LOG_TRACE("offset {} qend {}", offset, qend);

            if (cigar_start_idx >= 0) {
                const int32_t current_w_start = (t_window_start / window_size) * window_size;
                const int32_t current_w_end = std::min(current_w_start + window_size, tlen);
                windows[new_w - 1].push_back({aln_idx, current_w_start, current_w_end,
                                              t_window_start, tpos + offset, q_window_start, qend,
                                              cigar_start_idx, cigar_start_offset, cigar_end_idx,
                                              cigar_end_offset, 0.f, 0});
                LOG_TRACE(
                        "pushed t_window_start {} q_window_start {} qend {} cigar_start_idx {} "
                        "cigar_start_offseet {} cigar_end_idx {} cigar_end_offset {}",
                        t_window_start, q_window_start, qend, cigar_start_idx, cigar_start_offset,
                        cigar_end_idx, cigar_end_offset);

                t_window_start = tpos + offset;
                q_window_start = qend;
                cigar_start_idx = cigar_end_idx;
                cigar_start_offset = cigar_end_offset;
            } else {
                t_window_start = tpos + offset;
                q_window_start = qend;
                cigar_start_idx = cigar_end_idx;
                cigar_start_offset = cigar_end_offset;
            }

            tpos = tnew;
            qpos = qnew;
        }

        // Add the remaining portion of thar target read and query read/cigar into the last window.
        if (tpos > nth_window_thresh && (tpos % window_size != 0)) {
            const int32_t current_w_start = (t_window_start / window_size) * window_size;
            const int32_t current_w_end = std::min(current_w_start + window_size, tlen);
            windows[last_window - 1].push_back(
                    {aln_idx, current_w_start, current_w_end, t_window_start, tlen, q_window_start,
                     qpos, cigar_start_idx, cigar_start_offset, (int)cigar.size(), 0, 0.f, 0});
            LOG_TRACE(
                    "pushed t_window_start {} q_window_start {} qpos {} cigar_start_idx {} "
                    "cigar_start_offseet {} cigar len {} 0",
                    t_window_start, q_window_start, qpos, cigar_start_idx, cigar_start_offset,
                    cigar.size());
        }
    }

    return true;
}

std::vector<std::pair<int32_t, OverlapWindow>> split_alignment(
        const utils::Overlap& overlap,
        const std::vector<CigarOp>& cigar,
        const std::vector<secondary::Interval>& win_intervals,
        const int32_t aln_id,
        const bool legacy_qstart,
        const bool custom_initial_point,
        const int32_t initial_cig_id,
        const int32_t initial_cig_offset,
        const int32_t initial_tstart,
        const int32_t initial_qstart) {
    utils::ScopedProfileRange spr("split_alignment", 3);

    if (std::empty(cigar)) {
        return {};
    }

    if (std::empty(win_intervals)) {
        return {};
    }

    std::vector<std::pair<int32_t, OverlapWindow>> result;
    int32_t ref_pos = custom_initial_point ? initial_tstart : overlap.tstart;
    int32_t query_pos =
            custom_initial_point ? initial_qstart : (legacy_qstart ? 0 : overlap.qstart);
    int32_t cigar_idx = custom_initial_point ? initial_cig_id : 0;
    int32_t cigar_offset = custom_initial_point ? initial_cig_offset : 0;

    const int32_t num_cigars = static_cast<int32_t>(std::size(cigar));

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
    if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
        (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
        std::cerr << "[split_alignment entry] aln_id = " << aln_id
                  << ", custom_initial_point = " << custom_initial_point
                  << ", initial_tstart = " << initial_tstart
                  << ", initial_qstart = " << initial_qstart << ", ref_pos = " << ref_pos
                  << ", query_pos = " << query_pos << ", cigar_idx = " << cigar_idx
                  << ", cigar_offset = " << cigar_offset << ", overlap = {" << overlap << "}"
                  << "\n";
    }
#endif

    for (int32_t win_id = 0; win_id < static_cast<int32_t>(std::size(win_intervals)); ++win_id) {
        const auto& win = win_intervals[win_id];

        OverlapWindow win_data;
        win_data.overlap_idx = aln_id;
        win_data.win_tstart = win.start;
        win_data.win_tend = win.end;

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
        if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
            (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
            std::cerr << "[split_alignment win_id = " << win_id << ", aln_id = " << aln_id
                      << "] win.start = " << win.start << ", win.end = " << win.end
                      << ", ref_pos = " << ref_pos << ", query_pos = " << query_pos
                      << ", cigar_idx = " << cigar_idx << ", cigar_offset = " << cigar_offset
                      << "\n";
        }
#endif

        if (ref_pos >= win.end) {
#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
            if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                std::cerr << "    -> [win_id = " << win_id
                          << "] Continuing, ref_pos >= win.end. ref_pos = " << ref_pos
                          << ", win.end = " << win.end << "\n";
            }
#endif

            continue;
        }

        while (cigar_idx < num_cigars) {
            const auto& op = cigar[cigar_idx];
            const int32_t len = op.len;

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
            if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                std::cerr << "    [win_id = " << win_id << "] cigar_idx = " << cigar_idx
                          << ", cigar_offset = " << cigar_offset << ", num_cigars = " << num_cigars
                          << ", op = " << op << ", ref_pos = " << ref_pos
                          << ", query_pos = " << query_pos << ", aln_id = " << aln_id << "\n";
            }
#endif

            const bool consumes_target = ((op.op == CigarOpType::M) || (op.op == CigarOpType::EQ) ||
                                          (op.op == CigarOpType::X) || (op.op == CigarOpType::D) ||
                                          (op.op == CigarOpType::N));
            const bool consumes_query = ((op.op == CigarOpType::M) || (op.op == CigarOpType::EQ) ||
                                         (op.op == CigarOpType::X) || (op.op == CigarOpType::I) ||
                                         (op.op == CigarOpType::S));

            // Found window start.
            // Elaborated: if the op consumes the target and the window has not yet been initialized,
            // and this is the first operation which crosses the entrance into this window, set it.
            if (consumes_target && (win_data.tstart == -1) &&
                ((ref_pos + len - cigar_offset) > win.start)) {
#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
                if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                    (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                    std::cerr << "        - Start found. ref_pos = " << ref_pos
                              << ", query_pos = " << query_pos
                              << ", cigar_offset = " << cigar_offset << "\n";
                }
#endif
                const int32_t dist = std::max(0, win.start - ref_pos);
                win_data.tstart = std::max(ref_pos, win.start);
                win_data.qstart = consumes_query ? (query_pos + dist) : query_pos;
                win_data.cigar_start_idx = cigar_idx;
                win_data.cigar_start_offset = cigar_offset + dist;
                win_data.columns = 0;
                ref_pos = win_data.tstart;
                query_pos = win_data.qstart;
                cigar_offset = win_data.cigar_start_offset;
#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
                if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                    (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                    std::cerr << "            -> win_data = " << win_data << "\n";
                }
#endif
            }

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
            if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                std::cerr << "        - (between checks) ref_pos = " << ref_pos
                          << ", query_pos = " << query_pos << ", cigar_offset = " << cigar_offset
                          << ", win_data = {" << win_data << "}" << "\n";
            }
#endif

            // Found window end.
            // Elaborated: If this operation consumes the target, and the beginning of this op falls within
            // the current window, but the end of this op exits the current window, set the end positions.
            // Exit the loop once the window is fully processed.
            if (consumes_target && (ref_pos <= win.end) &&
                ((ref_pos + len - cigar_offset) > win.end)) {
#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
                if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                    (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                    std::cerr << "        - End found. ref_pos = " << ref_pos
                              << ", query_pos = " << query_pos
                              << ", cigar_offset = " << cigar_offset << "\n";
                }
#endif

                // The last CIGAR event is split on window boundary.
                const int32_t dist = std::max(0, win.end - ref_pos);
                win_data.tend = ref_pos + dist;
                win_data.qend = consumes_query ? (query_pos + dist) : query_pos;
                win_data.cigar_end_idx = cigar_idx;
                win_data.cigar_end_offset = cigar_offset + dist;
                win_data.columns += dist;
                cigar_offset = dist;
                if (win_data.cigar_end_offset == len) {
                    ++win_data.cigar_end_idx;
                    win_data.cigar_end_offset = 0;
                }
                ref_pos = win_data.tend;
                query_pos = win_data.qend;
                cigar_offset = win_data.cigar_end_offset;

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
                if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                    (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                    std::cerr << "            -> win_data = " << win_data << "\n";
                }
#endif

                break;
            }

            // Move.
            if (consumes_target) {
                ref_pos += len - cigar_offset;
            }
            if (consumes_query) {
                query_pos += len - cigar_offset;
            }

            // Update the target end coordinates, in case the alignment ends early.
            win_data.tend = ref_pos;
            win_data.qend = query_pos;
            win_data.cigar_end_idx = cigar_idx + 1;
            win_data.cigar_end_offset = 0;
            win_data.columns += (len - cigar_offset);

            ++cigar_idx;
            cigar_offset = 0;

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
            if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                std::cerr << "    - Next iter or end. ref_pos = " << ref_pos
                          << ", query_pos = " << query_pos << ", columns = " << win_data.columns
                          << ", cigar_offset = " << cigar_offset << "\n";
                std::cerr << "            -> win_data = " << win_data << "\n";
            }
#endif
        }

        // Add the new window data but only if the start coordinate was found
        // (i.e. the window is valid).
        if (win_data.tstart >= 0) {
            result.emplace_back(win_id, win_data);

#ifdef DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID
            if ((DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID < 0) ||
                (aln_id == DEBUG_CORRECTION_SPLIT_ALIGNMENT_ID)) {
                std::cerr << "[split_alignment] Emplacing: win_id = " << win_id << ", win_data = {"
                          << win_data << "}\n";
            }
#endif

            ref_pos = win_data.tend;
            query_pos = win_data.qend;
            cigar_idx = win_data.cigar_end_idx;
            cigar_offset = win_data.cigar_end_offset;
        }
    }
    return result;
}

std::vector<std::vector<OverlapWindow>> split_alignments_into_windows(
        const CorrectionAlignments& alignments,
        const std::vector<secondary::Interval>& win_intervals,
        const int32_t window_size,
        const std::vector<int32_t>& aln_ids,
        const std::vector<CigarPoint>& cigar_points) {
    utils::ScopedProfileRange spr("split_alignments_into_windows", 2);

    if (std::empty(win_intervals)) {
        return {};
    }

    const auto check_margin = [](const OverlapWindow& win, const int32_t zeroth_window_thresh,
                                 const int32_t nth_window_thresh) {
        if ((win.tstart > win.win_tstart) && (win.tstart >= zeroth_window_thresh)) {
            return false;
        }
        if ((win.tend < win.win_tend) && (win.tend <= nth_window_thresh)) {
            return false;
        }
        return true;
    };

    const int32_t num_alns = static_cast<int32_t>(std::size(alignments.overlaps));

    std::vector<std::vector<OverlapWindow>> windows(std::size(win_intervals));

    std::vector<int32_t> ids_to_process = aln_ids;
    if (std::empty(ids_to_process)) {
        ids_to_process.resize(num_alns);
        std::iota(std::begin(ids_to_process), std::end(ids_to_process), 0);
    }

    assert(std::empty(cigar_points) || (std::size(cigar_points) == std::size(alignments.cigars)));

    for (const int32_t aln_id : ids_to_process) {
        // Skip short overlaps.
        const auto& overlap = alignments.overlaps[aln_id];
        const auto& cigar = alignments.cigars[aln_id];
        const int32_t zeroth_window_thresh = static_cast<int32_t>(0.1f * window_size);
        const int32_t nth_window_thresh = overlap.tlen - zeroth_window_thresh;

        if ((overlap.tend - overlap.tstart < window_size) ||
            (overlap.qend - overlap.qstart < window_size)) {
            continue;
        }

        const CigarPoint cigar_start = (!std::empty(cigar_points))
                                               ? cigar_points[aln_id]
                                               : CigarPoint{0, 0, overlap.tstart, overlap.qstart};

        // Split the alignment into windows.
        const auto aln_windows =
                split_alignment(overlap, cigar, win_intervals, aln_id, true, true, cigar_start.idx,
                                cigar_start.offset, cigar_start.tpos, cigar_start.qpos);

        // Filter and assign windows.
        for (const auto& [win_id, win] : aln_windows) {
            if (!check_margin(win, zeroth_window_thresh, nth_window_thresh)) {
                continue;
            }
            assert(win_id < static_cast<int32_t>(std::size(win_intervals)));
            windows[win_id].emplace_back(win);
        }
    }

    return windows;
}

std::vector<secondary::Interval> extract_limited_windows(
        std::vector<std::vector<OverlapWindow>>& windows,
        const CorrectionAlignments& alignments,
        const int32_t window_size,
        const int32_t max_num_columns) {
    utils::ScopedProfileRange spr("extract_limited_windows", 1);

    windows.clear();

    if (std::empty(alignments.overlaps)) {
        return {};
    }

    const int32_t tlen = alignments.overlaps.front().tlen;

    std::vector<secondary::Interval> win_intervals;

    std::vector<CigarPoint> cigar_points(std::size(alignments.cigars));
    for (size_t i = 0; i < std::size(alignments.overlaps); ++i) {
        // Initialize only tpos and not qpos because of the design of feature extraction
        // which expects the query coordinate to start with 0 and not with actual qstart.
        cigar_points[i].tpos = alignments.overlaps[i].tstart;
    }

    // Incrementally create windows because each window can be of different size.
    int32_t win_start = 0;
    while (win_start < tlen) {
        const int32_t max_win_end = std::min(tlen, win_start + window_size);

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] Initial window: win_start = " << win_start
                  << ", max_win_end = " << max_win_end << ", tlen = " << tlen
                  << ", read_name = " << alignments.read_name << "\n";
        for (size_t i = 0; i < std::size(alignments.overlaps); ++i) {
            std::cerr << "[i = " << i << "] cigar_points[i] = {idx = " << cigar_points[i].idx
                      << ", offset = " << cigar_points[i].offset
                      << ", tpos = " << cigar_points[i].tpos << ", qpos = " << cigar_points[i].qpos
                      << ", cigar.size = " << std::size(alignments.cigars[i])
                      << ", qname = " << alignments.qnames[i] << ", overlap = {"
                      << alignments.overlaps[i] << "}" << "\n";
        }
#endif

        // Extract CIGARs for the current window (only ONE window) starting from the previous CIGAR locations (cigar_points).
        auto new_windows = split_alignments_into_windows(
                alignments, {secondary::Interval{win_start, max_win_end}}, window_size, {},
                cigar_points);

        // Sanity check. One input window was given, one output window is expected.
        if (std::size(new_windows) != 1) {
            spdlog::warn(
                    "Expected exactly 1 output window, but {} were generated after "
                    "split_alignments_into_windows!",
                    std::size(new_windows));
            windows.clear();
            return {};
        }

        // Update the cigar_points for the next window.
        // Since the window end position is still not determined and would require re-processing of the alignments,
        // we set the cigar_points to the beginning of the current window and process one extra window of data for
        // each new window (which is a linear overhead and fine; compared to processing the entire CIGAR every time).
        for (size_t i = 0; i < std::size(new_windows.front()); ++i) {
            const auto& w = new_windows.front()[i];
            cigar_points[w.overlap_idx].idx = w.cigar_start_idx;
            cigar_points[w.overlap_idx].offset = w.cigar_start_offset;
            cigar_points[w.overlap_idx].tpos = w.tstart;
            cigar_points[w.overlap_idx].qpos = w.qstart;
        }

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] new_windows[0].size = " << std::size(new_windows[0])
                  << "\n";
#endif

        // This: (1) filters overlaps with large indels, (2) computes accuracy, (3) sorts and (4) picks TOP_K overlaps.
        // It updates the new_windows in place.
        const std::unordered_set<int32_t> overlap_idxs = filter_features(new_windows, alignments);
        if (overlap_idxs.empty()) {
            windows.emplace_back(std::vector<OverlapWindow>());
            win_intervals.emplace_back(secondary::Interval{win_start, max_win_end});
            win_start = max_win_end;
            continue;
        }

        // The new_windows are updated by filter_features.
        // Sanity check. There should still be just one window.
        if (std::size(new_windows) != 1) {
            spdlog::warn(
                    "Expected exactly 1 output window, but {} were generated after "
                    "filter_features!",
                    std::size(new_windows));
            windows.clear();
            return {};
        }

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] Filtered windows, new_windows[0].size = "
                  << std::size(new_windows[0]) << "\n";
#endif

        const std::vector<int32_t> consumed =
                calc_max_step_per_pos(alignments, new_windows.front(), window_size, 1);

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] find_target_end_for_window"
                  << ", win_start = " << win_start << ", max_win_end = " << max_win_end
                  << ", max_num_columns = " << max_num_columns << "\n";
#endif

        // Compute the new window end coordinate which limits the number of columns.
        const int32_t new_win_end =
                find_target_end_for_window(consumed, win_start, max_win_end, max_num_columns);

        win_intervals.emplace_back(secondary::Interval{win_start, new_win_end});

        // Collect the IDs of the alignments which will be used in the end.
        std::vector<int32_t> aln_ids;
        for (const auto& win : new_windows.front()) {
            aln_ids.emplace_back(win.overlap_idx);
        }

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] Recomputing the alignment splits after computed "
                     "new_win_end = "
                  << new_win_end << "\n";
#endif

        // Compute the new slices but only for the alignments which survived initial filtering.
        new_windows = split_alignments_into_windows(alignments,
                                                    {secondary::Interval{win_start, new_win_end}},
                                                    window_size, aln_ids, cigar_points);
        if (std::size(new_windows) != 1) {
            spdlog::warn(
                    "Expected exactly 1 output window, but {} were generated after second "
                    "split_alignments_into_windows!",
                    std::size(new_windows));
            windows.clear();
            return {};
        }

        // Finally, emplace the new windows.
        windows.emplace_back(std::move(new_windows.front()));

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
        std::cerr << "[extract_limited_windows] Emplaced new interval: [" << win_start << ", "
                  << new_win_end << "], win_intervals.size = " << std::size(win_intervals)
                  << "\n\n";
#endif

        win_start = new_win_end;
    }

    if (win_start < tlen) {
        win_intervals.emplace_back(secondary::Interval{win_start, tlen});
        windows.emplace_back(std::vector<OverlapWindow>());
        std::cerr << "[extract_limited_windows] Emplaced new final interval: [" << win_start << ", "
                  << tlen << "]\n";
    }

#ifdef DEBUG_CORRECT_EXTRACT_LIMITED_WINDOWS_2
    std::cerr << "[extract_limited_windows] win_intervals.size = " << win_intervals.size() << "\n";
    for (size_t i = 0; i < std::size(win_intervals); ++i) {
        std::cerr << "[win_interval i = " << i << "] start = " << win_intervals[i].start
                  << ", end = " << win_intervals[i].end << "\n";
    }
#endif

    return win_intervals;
}

}  // namespace dorado::correction
