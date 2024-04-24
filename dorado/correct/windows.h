#pragma once

#include "conversions.h"
#include "types.h"

namespace dorado::correction {

void extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                     const CorrectionAlignments& alignments,
                     int m_window_size) {
    size_t num_alignments = alignments.overlaps.size();
    for (size_t a = 0; a < num_alignments; a++) {
        const auto& overlap = alignments.overlaps[a];
        const auto& cigar = alignments.cigars[a];
        //if (alignments.qnames[a] != "e3066d3e-2bdf-4803-89b9-0f077ac7ff7f") {
        //    continue;
        //}
        //spdlog::info("window for {}", alignments.qnames[a]);
        //const std::string& qseq = alignments.seqs[a];

        // Following the is_target == False logic form the rust code.
        if (overlap.tend - overlap.tstart < m_window_size) {
            continue;
        }

        //spdlog::info("qlen {} qstart {} qend {} strand {} tlen {} tstart {} tend {}", overlap.qlen,
        //             overlap.qstart, overlap.qend, overlap.fwd, overlap.tlen, overlap.tstart,
        //             overlap.tend);

        int first_window = -1;
        int last_window = -1;
        int tstart = -1;
        int tpos = -1;
        int qpos = 0;

        int zeroth_window_thresh = (0.1f * m_window_size);
        int nth_window_thresh = overlap.tlen - zeroth_window_thresh;

        //spdlog::info("zeroth {} nth {}", zeroth_window_thresh, nth_window_thresh);

        first_window = (overlap.tstart < zeroth_window_thresh
                                ? 0
                                : (overlap.tstart + m_window_size - 1) / m_window_size);
        last_window = (overlap.tend > nth_window_thresh ? (overlap.tend - 1) / m_window_size + 1
                                                        : overlap.tend / m_window_size);
        tstart = overlap.tstart;
        tpos = overlap.tstart;

        //spdlog::info("first window {} last window {} tstart {} tpos {}", first_window, last_window,
        //             tstart, tpos);

        if (last_window - first_window < 1) {
            continue;
        }

        int t_window_start = -1;
        int q_window_start = -1;
        int cigar_start_idx = -1;
        int cigar_start_offset = -1;

        //spdlog::info("tpos {} qpos {}", tpos, qpos);

        if ((tpos % m_window_size == 0) || (tstart < zeroth_window_thresh)) {
            t_window_start = tpos;
            q_window_start = qpos;
            cigar_start_idx = 0;
            cigar_start_offset = 0;
        }

        //spdlog::info("t_window_start {} q_window_start {} cigar_start_idx {} cigar_start_offset {}",
        //             t_window_start, q_window_start, cigar_start_idx, cigar_start_offset);

        for (size_t cigar_idx = 0; cigar_idx < cigar.size(); cigar_idx++) {
            auto op = cigar[cigar_idx];
            int tnew = tpos;
            int qnew = qpos;
            switch (op.op) {
            case CigarOpType::MATCH:
            case CigarOpType::MISMATCH:
                tnew = tpos + op.len;
                qnew = qpos + op.len;
                //spdlog::info("{} {}", op.len, "M");
                break;
            case CigarOpType::DEL:
                tnew = tpos + op.len;
                //spdlog::info("{} {}", op.len, "D");
                break;
            case CigarOpType::INS:
                qpos += op.len;
                //spdlog::info("{} {}", op.len, "I");
                continue;
            default:
                continue;
            }

            //spdlog::info("tpos {} qpos {} tnew {} qnew {}", tpos, qpos, tnew, qnew);

            int current_w = tpos / m_window_size;
            int new_w = tnew / m_window_size;
            int diff_w = new_w - current_w;

            if (diff_w == 0) {
                tpos = tnew;
                qpos = qnew;
                continue;
            }

            for (int i = 1; i < diff_w; i++) {
                int offset = (current_w + i) * m_window_size - tpos;

                int q_start_new = (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH)
                                          ? qpos + offset
                                          : qpos;

                if (cigar_start_idx >= 0) {
                    windows[(current_w + i) - 1].push_back(
                            {a, t_window_start, q_window_start, q_start_new, cigar_start_idx,
                             cigar_start_offset, (int)cigar_idx, offset});

                    //spdlog::info("pushed t_window_start {} q_window_start {} q_start_new {} cigar_start_idx {} cigar_start_offseet {} cigar_idx {} offset {}", t_window_start, q_window_start, q_start_new, cigar_start_idx, cigar_start_offset, cigar_idx, offset);

                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }

                    cigar_start_idx = cigar_idx;
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

            //spdlog::info("new_w {} window size {} tpos {}", new_w, m_window_size, tpos);
            int offset = new_w * m_window_size - tpos;

            int qend = (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH)
                               ? qpos + offset
                               : qpos;

            //spdlog::info("offset {} qend {}", offset, qend);

            int cigar_end_idx = -1;
            int cigar_end_offset = -1;

            if (tnew == new_w * m_window_size) {
                if (cigar_idx + 1 < cigar.size() && cigar[cigar_idx + 1].op == CigarOpType::INS) {
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

            //spdlog::info("offset {} qend {}", offset, qend);

            if (cigar_start_idx >= 0) {
                windows[new_w - 1].push_back({a, t_window_start, q_window_start, qend,
                                              cigar_start_idx, cigar_start_offset, cigar_end_idx,
                                              cigar_end_offset});
                //spdlog::info("pushed t_window_start {} q_window_start {} qend {} cigar_start_idx {} cigar_start_offseet {} cigar_end_idx {} cigar_end_offset {}", t_window_start, q_window_start, qend, cigar_start_idx, cigar_start_offset, cigar_end_idx, cigar_end_offset);

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

        if (tpos > nth_window_thresh && (tpos % m_window_size != 0)) {
            windows[last_window - 1].push_back({a, t_window_start, q_window_start, qpos,
                                                cigar_start_idx, cigar_start_offset,
                                                (int)cigar.size(), 0});
            //spdlog::info("pushed t_window_start {} q_window_start {} qpos {} cigar_start_idx {} cigar_start_offseet {} cigar len {} 0", t_window_start, q_window_start, qpos, cigar_start_idx, cigar_start_offset, cigar.size());
        }
    }
}
}  // namespace dorado::correction
