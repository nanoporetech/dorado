#include "windows.h"

#include "conversions.h"
#include "read_pipeline/messages.h"
#include "types.h"

#include <spdlog/spdlog.h>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::correction {

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
    int num_alignments = (int)alignments.overlaps.size();
    for (int aln_idx = 0; aln_idx < num_alignments; aln_idx++) {
        const auto& overlap = alignments.overlaps[aln_idx];
        const auto& cigar = alignments.cigars[aln_idx];
        //if (alignments.qnames[a] != "e3066d3e-2bdf-4803-89b9-0f077ac7ff7f") {
        //    continue;
        //}
        LOG_TRACE("window for {}", alignments.qnames[aln_idx]);

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
            spdlog::debug("{} num windows {} for read {} with readlen {}", alignments.read_name,
                          windows.size(), alignments.read_name, alignments.read_seq.length());
            spdlog::debug(
                    "{} zeoth thres {} nth thres {} first win {} last win {} windows size {} "
                    "overlap "
                    "tlen {} overlsp tstart {} overlap tend {} qname {} qlen {} qstart {} qend {}",
                    alignments.read_name, zeroth_window_thresh, nth_window_thresh, first_window,
                    last_window, windows.size(), overlap.tlen, overlap.tstart, overlap.tend,
                    alignments.qnames[aln_idx], overlap.qlen, overlap.qstart, overlap.qend);
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
            case CigarOpType::MATCH:
            case CigarOpType::MISMATCH:
                tnew = tpos + op.len;
                qnew = qpos + op.len;
                LOG_TRACE("{} {}", op.len, "M");
                break;
            case CigarOpType::DEL:
                tnew = tpos + op.len;
                LOG_TRACE("{} {}", op.len, "D");
                break;
            case CigarOpType::INS:
                qpos += op.len;
                LOG_TRACE("{} {}", op.len, "I");
                continue;
            default:
                continue;
            }

            LOG_TRACE("tpos {} qpos {} tnew {} qnew {}", tpos, qpos, tnew, qnew);

            int current_w = tpos / window_size;
            int new_w = tnew / window_size;
            int diff_w = new_w - current_w;

            if (diff_w == 0) {
                tpos = tnew;
                qpos = qnew;
                continue;
            }

            // If the cigar operations spans multiple windows, break it up into
            // multiple chunks with intervals corresponding to each window.
            for (int i = 1; i < diff_w; i++) {
                int offset = (current_w + i) * window_size - tpos;

                int q_start_new = (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH)
                                          ? qpos + offset
                                          : qpos;

                if (cigar_start_idx >= 0) {
                    windows[(current_w + i) - 1].push_back(
                            {aln_idx, t_window_start, q_window_start, q_start_new, cigar_start_idx,
                             cigar_start_offset, (int)cigar_idx, offset, 0.f});

                    LOG_TRACE(
                            "pushed t_window_start {} q_window_start {} q_start_new {} "
                            "cigar_start_idx {} cigar_start_offseet {} cigar_idx {} offset {}",
                            t_window_start, q_window_start, q_start_new, cigar_start_idx,
                            cigar_start_offset, cigar_idx, offset);

                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }

                    cigar_start_idx = (int)cigar_idx;
                    cigar_start_offset = offset;
                } else {
                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }

                    cigar_start_idx = cigar_idx;
                    cigar_start_offset = offset;
                }
            }

            LOG_TRACE("new_w {} window size {} tpos {}", new_w, window_size, tpos);
            int offset = new_w * window_size - tpos;

            int qend = (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH)
                               ? qpos + offset
                               : qpos;

            LOG_TRACE("offset {} qend {}", offset, qend);

            int cigar_end_idx = -1;
            int cigar_end_offset = -1;

            if (tnew == new_w * window_size) {
                if (cigar_idx + 1 < (int)cigar.size() &&
                    cigar[cigar_idx + 1].op == CigarOpType::INS) {
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
                windows[new_w - 1].push_back({aln_idx, t_window_start, q_window_start, qend,
                                              cigar_start_idx, cigar_start_offset, cigar_end_idx,
                                              cigar_end_offset, 0.f});
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
            windows[last_window - 1].push_back({aln_idx, t_window_start, q_window_start, qpos,
                                                cigar_start_idx, cigar_start_offset,
                                                (int)cigar.size(), 0, 0.f});
            LOG_TRACE(
                    "pushed t_window_start {} q_window_start {} qpos {} cigar_start_idx {} "
                    "cigar_start_offseet {} cigar len {} 0",
                    t_window_start, q_window_start, qpos, cigar_start_idx, cigar_start_offset,
                    cigar.size());
        }
    }

    return true;
}

}  // namespace dorado::correction
