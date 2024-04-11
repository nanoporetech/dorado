#include "CorrectionNode.h"

#include "ClientInfo.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2IndexSupportTypes.h"
#include "alignment/Minimap2Options.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

//void print_size(const torch::Tensor& t, const std::string& name) {
//    std::string size = "";
//    for (auto s : t.sizes()) {
//        size += std::to_string(s) + ",";
//    }
//    std::stringstream ss;
//    ss << t.dtype();
//    spdlog::info("{} tensor size {} dtype {}", name, size, ss.str());
//}

}  // namespace

namespace dorado {

const int TOP_K = 30;

struct OverlapWindow {
    size_t overlap_idx = -1;
    int tstart = -1;
    int qstart = -1;
    int qend = -1;
    int cigar_start_idx = -1;
    int cigar_start_offset = -1;
    int cigar_end_idx = -1;
    int cigar_end_offset = -1;
    float accuracy = 0;
};

struct WindowFeatures {
    torch::Tensor bases;
    torch::Tensor quals;
    std::vector<torch::Tensor> indices;
    torch::Tensor length;
    std::vector<std::pair<int, int>> supported;
    std::vector<char> inferred_bases;
    int n_alns = 0;
};

std::array<int, 128> base_forward_mapping() {
    std::array<int, 128> base_forward = {0};
    base_forward['*'] = '*';
    base_forward['#'] = '*';
    base_forward['A'] = 'A';
    base_forward['a'] = 'A';
    base_forward['T'] = 'T';
    base_forward['t'] = 'T';
    base_forward['C'] = 'C';
    base_forward['c'] = 'C';
    base_forward['G'] = 'G';
    base_forward['g'] = 'G';
    return base_forward;
}

std::array<int, 128> gen_base_encoding() {
    std::array<int, 128> base_encoding = {0};
    const std::string bases = "ACGT*acgt#.";
    for (size_t i = 0; i < bases.length(); i++) {
        base_encoding[bases[i]] = i;
    }
    return base_encoding;
}

std::array<int, 11> gen_base_decoding() {
    std::array<int, 11> base_decoding = {0};
    const std::string bases = "ACGT*acgt#.";
    for (size_t i = 0; i < bases.length(); i++) {
        base_decoding[i] = bases[i];
    }
    return base_decoding;
}

bool filter_overlap(const OverlapWindow& overlap, const CorrectionAlignments& alignments) {
    bool long_indel = false;
    const auto& cigar = alignments.cigars[overlap.overlap_idx];
    for (size_t i = overlap.cigar_start_idx;
         i < std::min(size_t(overlap.cigar_end_idx + 1), cigar.size()); i++) {
        if (cigar[i].op == CigarOpType::INS || cigar[i].op == CigarOpType::DEL) {
            long_indel |= cigar[i].len >= 30;
        }
    }
    return long_indel;
}

void calculate_accuracy(OverlapWindow& overlap,
                        const CorrectionAlignments& alignments,
                        size_t win_idx,
                        int win_len,
                        int m_window_size) {
    int tstart = overlap.tstart;
    int tend = win_idx * m_window_size + win_len;

    // get query region
    const auto overlap_idx = overlap.overlap_idx;
    int oqstart = alignments.overlaps[overlap_idx].qstart;
    int oqend = alignments.overlaps[overlap_idx].qend;
    int qstart, qend;
    if (alignments.overlaps[overlap_idx].fwd) {
        qstart = oqstart + overlap.qstart;
        qend = oqstart + overlap.qend;
    } else {
        qstart = oqend - overlap.qend;
        qend = oqend - overlap.qstart;
    }

    int qlen = qend - qstart;

    // Fetch subsequences
    std::string tseq = alignments.read_seq.substr(tstart, tend - tstart);
    std::string qseq;
    if (alignments.overlaps[overlap_idx].fwd) {
        qseq = alignments.seqs[overlap_idx].substr(qstart, qlen);
    } else {
        qseq = utils::reverse_complement(alignments.seqs[overlap_idx].substr(qstart, qlen));
    }

    //spdlog::info("tstart {} tend {} qstart {} qend {} cig st {} cig end {}", tstart, tend, qstart,
    //             qend, overlap.cigar_start_idx, overlap.cigar_end_idx);

    const auto& cigar = alignments.cigars[overlap.overlap_idx];

    // Calculate accuracy
    int tpos = 0, qpos = 0;
    int m = 0, s = 0, i = 0, d = 0;

    for (int idx = overlap.cigar_start_idx; idx <= overlap.cigar_end_idx; idx++) {
        int len = -1;
        if (overlap.cigar_start_idx == overlap.cigar_end_idx) {
            len = overlap.cigar_end_offset - overlap.cigar_start_offset;
        } else if (idx == overlap.cigar_start_idx) {
            len = (cigar[idx].len - overlap.cigar_start_offset);
        } else if (idx == overlap.cigar_end_idx) {
            len = overlap.cigar_end_offset;
        } else {
            len = cigar[idx].len;
        }

        if (len == 0) {
            break;
        }

        //spdlog::info("len {} tpos {} qpos {}", len, tpos, qpos);

        switch (cigar[idx].op) {
        case CigarOpType::MATCH:
            for (int j = 0; j < len; j++) {
                auto tbase = tseq[tpos + j];
                auto qbase = qseq[qpos + j];
                //spdlog::info("{} tbase {}, {} qbase {}", tpos + j, tbase, qpos + j, qbase);

                if (tbase == qbase) {
                    m += 1;
                } else {
                    s += 1;
                }
            }

            tpos += len;
            qpos += len;
            break;
        case CigarOpType::INS:
            i += len;
            qpos += len;
            break;
        case CigarOpType::DEL:
            d += len;
            tpos += len;
            break;
        default:
            break;
        }
    }

    //spdlog::info("m {} s {} i {} d {}", m, s, i, d);

    overlap.accuracy = (static_cast<float>(m) / (m + s + i + d));
    //spdlog::info("accuracy {}", overlap.accuracy);
}

std::vector<int> get_max_ins_for_window(const std::vector<OverlapWindow>& windows,
                                        const CorrectionAlignments& alignments,
                                        int tstart,
                                        int win_len) {
    std::vector<int> max_ins(win_len, 0);
    ;
    for (const auto& overlap : windows) {
        auto tpos = overlap.tstart - tstart;

        const auto& cigar = alignments.cigars[overlap.overlap_idx];
        int cigar_len = overlap.cigar_end_idx - overlap.cigar_start_idx + 1;

        for (int i = overlap.cigar_start_idx;
             i <= std::min(overlap.cigar_end_idx, int(cigar.size()) - 1); i++) {
            CigarOpType op = cigar[i].op;
            int len = cigar[i].len;

            int l = -1;
            if (op == CigarOpType::MATCH || op == CigarOpType::DEL) {
                l = len;
            } else if (op == CigarOpType::INS) {
                max_ins[tpos - 1] = std::max(len, max_ins[tpos - 1]);
                continue;
            }

            if (cigar_len == 1) {
                tpos += overlap.cigar_end_offset - overlap.cigar_start_offset;
            } else if (i == overlap.cigar_start_idx) {
                tpos += l - overlap.cigar_start_offset;
            } else if (i == overlap.cigar_end_idx) {
                tpos += overlap.cigar_end_offset;
            } else {
                tpos += l;
            }
        }
    }

    int sum = 0;
    for (auto i : max_ins) {
        sum += i;
    }
    //spdlog::info("max ins sum {}", sum);
    return max_ins;
}

std::tuple<torch::Tensor, torch::Tensor> get_features_for_window(
        const std::vector<OverlapWindow>& windows,
        const CorrectionAlignments& alignments,
        int win_len,
        int tstart,
        const std::vector<int>& max_ins) {
    static auto base_encoding = gen_base_encoding();
    auto bases_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto quals_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    int length = std::accumulate(max_ins.begin(), max_ins.end(), 0) + (int)max_ins.size();
    auto bases = torch::empty({length, 1 + TOP_K}, bases_options);
    bases.fill_(base_encoding['.']);
    auto quals = torch::empty({length, 1 + TOP_K}, quals_options);
    quals.fill_((float)'!');

    // Write target for window
    const std::string& tseq = alignments.read_seq;
    const std::vector<uint8_t>& tqual = alignments.read_qual;

    int tpos = 0;
    auto target_tbases_tensor = bases.index({torch::indexing::Slice(), 0});
    target_tbases_tensor.fill_(base_encoding['*']);
    auto target_tquals_tensor = quals.index({torch::indexing::Slice(), 0});
    for (int i = 0; i < win_len; i++) {
        target_tbases_tensor[tpos] = base_encoding[tseq[i + tstart]];
        target_tquals_tensor[tpos] = float(tqual[i + tstart] + 33);

        //spdlog::info("tpos {} base {} qual {}", tpos, tbases_tensor[tpos].item<uint8_t>(), tquals_tensor[tpos].item<float>());
        tpos += 1 + max_ins[i];
    }

    //std::exit(0);

    // Write bases for each overlap window
    for (int w = 0; w < (int)windows.size(); w++) {
        auto tbases_tensor = bases.index({torch::indexing::Slice(), w + 1});
        auto tquals_tensor = quals.index({torch::indexing::Slice(), w + 1});
        const auto& overlap = windows[w];
        const auto& cigar = alignments.cigars[overlap.overlap_idx];
        int offset = overlap.tstart - tstart;

        bool fwd = alignments.overlaps[overlap.overlap_idx].fwd;
        int oqstart = alignments.overlaps[overlap.overlap_idx].qstart;
        int oqend = alignments.overlaps[overlap.overlap_idx].qend;
        (void)oqend;

        int qstart = -1, qend = -1, qlen = -1;
        if (fwd) {
            qstart = oqstart + overlap.qstart;
            qend = oqstart + overlap.qend;
            qlen = qend - qstart;
        } else {
            qstart = oqend - overlap.qend;
            qend = oqend - overlap.qstart;
            qlen = qend - qstart;
        }

        //spdlog::info("qstart {} qend {} aln qstart {} aln qend {} overlap qstart {} overlap qend {}", qstart, qend, oqstart, oqend, overlap.qstart, overlap.qend);
        (void)qlen;
        int query_iter = 0;
        std::string qseq = alignments.seqs[overlap.overlap_idx].substr(qstart, qlen);
        std::vector<uint8_t> qqual(alignments.quals[overlap.overlap_idx].begin() + qstart,
                                   alignments.quals[overlap.overlap_idx].begin() + qend);
        if (!fwd) {
            qseq = utils::reverse_complement(qseq);
            std::reverse(qqual.begin(), qqual.end());
        }
        int cigar_len = overlap.cigar_end_idx - overlap.cigar_start_idx + 1;
        int cigar_end = std::min((int)cigar.size(), cigar_len);

        uint8_t gap = fwd ? '*' : '#';

        tbases_tensor.fill_(base_encoding[gap]);

        tpos = offset;
        int idx = offset + std::accumulate(max_ins.begin(), max_ins.begin() + offset, 0);

        //spdlog::info("cigar_len {}, cigar_end {}, gap {}, tpos {}, idx {}, fwd {}", cigar_len, cigar_end, gap, tpos, idx, fwd ? '+' : '-');

        if (idx > 0) {
            auto no_alignment = tbases_tensor.index({torch::indexing::Slice(0, idx)});
            no_alignment.fill_(base_encoding['.']);
        }

        for (int cigar_idx = 0; cigar_idx < cigar_end; cigar_idx++) {
            auto cigar_op = cigar[cigar_idx + overlap.cigar_start_idx];
            auto l = cigar_op.len;
            auto op = cigar_op.op;

            if (cigar_len == 1) {
                l = overlap.cigar_end_offset - overlap.cigar_start_offset;
            } else if (cigar_idx == 0) {
                l -= overlap.cigar_start_offset;
            } else if (cigar_idx == cigar_len - 1) {
                l = overlap.cigar_end_offset;
            }

            //spdlog::info("cigar_idx {} l {}", cigar_idx, l);

            switch (op) {
            case CigarOpType::MATCH:
            case CigarOpType::MISMATCH:
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    tbases_tensor[idx] = base;
                    tquals_tensor[idx] = (float)qual + 33;

                    //spdlog::info("idx {} base {}, qual {}", idx, tbases_tensor[idx].item<uint8_t>(), tquals_tensor[idx].item<float>());

                    idx += 1 + max_ins[tpos + i];
                    query_iter++;
                }

                tpos += l;
                break;
            case CigarOpType::DEL:
                for (uint32_t i = 0; i < l; i++) {
                    //spdlog::info("idx {}", idx);
                    idx += 1 + max_ins[tpos + i];
                }
                tpos += l;
                break;
            case CigarOpType::INS:
                idx -= max_ins[tpos - 1];
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    tbases_tensor[idx + i] = base;
                    tquals_tensor[idx + i] = (float)qual + 33;

                    //spdlog::info("idx + i {} base {}, qual {}", idx + i, tbases_tensor[idx + i].item<uint8_t>(), tquals_tensor[idx + i].item<float>());

                    query_iter++;
                }

                idx += max_ins[tpos - 1];
            }
        }

        if (idx < tbases_tensor.sizes()[0]) {
            auto no_alignment =
                    tbases_tensor.index({torch::indexing::Slice(idx, torch::indexing::None)});
            no_alignment.fill_(base_encoding['.']);
        }

        //std::exit(0);
    }

    return {bases, quals};
}

std::vector<std::pair<int, int>> get_supported(torch::Tensor& bases) {
    std::vector<std::pair<int, int>> supported;

    static auto base_forward = base_forward_mapping();
    static auto base_encoding = gen_base_encoding();
    static auto base_decoding = gen_base_decoding();

    const int cols = bases.sizes()[0];
    const int rows = bases.sizes()[1];

    //spdlog::info("cols {} rows {}", cols, rows);

    int tpos = -1, ins = 0;
    std::array<int, 128> counter;
    for (int c = 0; c < cols; c++) {
        if (bases[c][0].item<uint8_t>() == base_encoding['*']) {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }
        counter.fill(0);
        for (int r = 0; r < rows; r++) {
            auto base = bases[c][r].item<uint8_t>();
            //spdlog::info("row {} base {}", r, base);
            if (base == base_encoding['.']) {
                continue;
            }
            //spdlog::info("decoded base {}", base_decoding[base]);
            counter[base_forward[base_decoding[base]]]++;
            //spdlog::info("base {} pos {} counter {}", base, pos, counter[pos]);
        }

        //spdlog::info("col {} A {} C {} T {} G {} * {}", c, counter['A'], counter['C'], counter['T'], counter['G'], counter['*']);
        int count = std::count_if(counter.begin(), counter.end(), [](int num) { return num >= 3; });
        //spdlog::info("count {}", count);
        if (count >= 2) {
            supported.push_back({tpos, ins});
            //spdlog::info("support added for {} {}", tpos, ins);
        }
        //spdlog::info("num supported {}", supported.size());
    }
    return supported;
}

std::vector<int> get_indices(torch::Tensor bases, std::vector<std::pair<int, int>>& supported) {
    static auto base_encoding = gen_base_encoding();
    auto tbase_tensor = bases.index({torch::indexing::Slice(), 0});
    std::vector<int> indices;
    for (int i = 0; i < tbase_tensor.sizes()[0]; i++) {
        if (tbase_tensor[i].item<uint8_t>() != base_encoding['*']) {
            indices.push_back(i);
        }
    }

    std::vector<int> supported_indices;
    supported_indices.reserve(supported.size());
    for (auto [pos, ins] : supported) {
        supported_indices.push_back(indices[pos] + ins);
    }

    return supported_indices;
}

std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments,
                                             int m_window_size) {
    const std::string& tseq = alignments.read_seq;
    int tlen = tseq.length();

    std::vector<WindowFeatures> wfs;

    for (size_t w = 0; w < windows.size(); w++) {
        //for (size_t w = 0; w < 1; w++) {
        int win_len = (w == windows.size() - 1) ? tlen - m_window_size * w : m_window_size;
        //spdlog::info("win idx {}: win len {}", w, win_len);
        auto& overlap_windows = windows[w];

        // Filter overlaps with very large indels
        std::vector<OverlapWindow> filtered_overlaps;
        for (auto& ovlp : overlap_windows) {
            if (!filter_overlap(ovlp, alignments)) {
                filtered_overlaps.push_back(std::move(ovlp));
            }
        }
        spdlog::info("window {} pre filter windows {} post filter windows {}", w,
                     overlap_windows.size(), filtered_overlaps.size());
        windows[w] = std::move(filtered_overlaps);

        // Sort overlaps by score
        for (auto& ovlp : windows[w]) {
            calculate_accuracy(ovlp, alignments, w, win_len, m_window_size);
        }
        // Sort the filtered overlaps by accuracy score
        std::sort(windows[w].begin(), windows[w].end(),
                  [](const OverlapWindow& a, const OverlapWindow& b) {
                      return a.accuracy > b.accuracy;
                  });
        windows[w].resize(std::min(TOP_K, (int)windows[w].size()));

        if (windows[w].size() > 0) {
            spdlog::info("window {} 1st {}", w, windows[w][0].qend);
        }
        if (windows[w].size() > 1) {
            spdlog::info("window {} 1st {} 2nd {}", w, windows[w][0].qend, windows[w][1].qend);
        }

        WindowFeatures wf;
        wf.n_alns = (int)windows[w].size();
        if (windows[w].size() > 0) {
            // Find the maximum insert size
            auto max_ins =
                    get_max_ins_for_window(windows[w], alignments, w * m_window_size, win_len);

            // Create tensors
            auto [bases, quals] = get_features_for_window(windows[w], alignments, win_len,
                                                          w * m_window_size, max_ins);
            auto supported = get_supported(bases);
            spdlog::info("num supported {}", supported.size());

            wf.bases = std::move(bases);
            wf.quals = std::move(quals);
            wf.supported = std::move(supported);
            wf.length = std::move(
                    torch::full({1}, (int)wf.supported.size(),
                                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)));
            auto supported_indices = get_indices(wf.bases, wf.supported);

            wf.indices.push_back(
                    torch::from_blob(
                            supported_indices.data(), {(int)supported_indices.size()},
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                            .clone());
        }
        wfs.push_back(std::move(wf));
    }

    return wfs;
}

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
        spdlog::info("window for {}", alignments.qnames[a]);
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
                } else {
                    t_window_start = tpos + offset;

                    if (op.op == CigarOpType::MATCH || op.op == CigarOpType::MISMATCH) {
                        q_window_start = qpos + offset;
                    } else {
                        q_window_start = qpos;
                    }
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

WindowFeatures run_inference(torch::jit::script::Module& module, std::vector<WindowFeatures>& wfs) {
    torch::NoGradGuard no_grad;
    module.eval();

    auto decode_preds = [](const torch::Tensor& preds) {
        std::vector<char> bases;
        bases.reserve(preds.sizes()[0]);
        static std::array<char, 5> decoder = {'A', 'C', 'G', 'T', '*'};
        for (int i = 0; i < preds.sizes()[0]; i++) {
            auto base_idx = preds[i].item<int>();
            bases.push_back(decoder[base_idx]);
            //spdlog::info("{} decoded to {}", i, bases.back());
        }
        return bases;
    };
    for (auto& wf : wfs) {
        if (wf.n_alns > 1 && wf.supported.size() > 0) {
            wf.bases = wf.bases.unsqueeze(0);  //.to(torch::kInt32);
            wf.quals = 2.f * (wf.quals.unsqueeze(0) - 33.f) / (126.f - 33.f) - 1.f;
            ;
            //print_size(wf.bases, "bases");
            //print_size(wf.quals, "quals");
            //print_size(wf.length, "length");
            //print_size(wf.indices[0], "indices");

            //spdlog::info("bases max {} min {}", wf.bases.max().item<uint8_t>(), wf.bases.min().item<uint8_t>());
            //spdlog::info("quals max {} min {}", wf.quals.max().item<float>(), wf.quals.min().item<float>());

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(wf.bases);
            inputs.push_back(wf.quals);
            inputs.push_back(wf.length);
            inputs.push_back(wf.indices);

            auto output = module.forward(inputs);
            if (output.isTuple()) {
                auto base_logits = output.toTuple()->elements()[1].toTensor();
                auto preds = base_logits.argmax(1, false);
                //print_size(base_logits, "base_logits");
                //print_size(preds, "preds");
                wf.inferred_bases = decode_preds(preds);
            }
        }
    }

    WindowFeatures f;
    return f;
}

struct base_count_t {
    int c = 0;
    char b;
};

std::vector<std::string> decode_windows(const std::vector<WindowFeatures>& wfs) {
    std::vector<std::string> corrected_reads;
    std::string corrected_seq;

    auto base_to_idx_map = []() {
        std::array<int, 128> map = {0};
        map['A'] = 0;
        map['C'] = 1;
        map['G'] = 2;
        map['T'] = 3;
        map['*'] = 4;
        map['a'] = 0;
        map['c'] = 1;
        map['g'] = 2;
        map['t'] = 3;
        map['#'] = 4;
        return map;
    };
    static auto base_to_idx = base_to_idx_map();
    static auto base_decoding = gen_base_decoding();
    static auto base_forward = base_forward_mapping();

    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
        }
    };
    struct PairEqual {
        bool operator()(const std::pair<int, int>& p1, const std::pair<int, int>& p2) const {
            return p1.first == p2.first && p1.second == p2.second;
        }
    };

    std::unordered_map<std::pair<int, int>, char, PairHash, PairEqual> bases_map;

    for (const auto& wf : wfs) {
        if (wf.n_alns < 2) {
            if (corrected_seq.length() > 0) {
                corrected_reads.push_back(corrected_seq);
                spdlog::info("added seq naln < 2 of len {}", corrected_seq.length());
                corrected_seq = "";
            }
            continue;
        }

        bases_map.clear();
        for (size_t i = 0; i < wf.supported.size(); i++) {
            //spdlog::info("supported positions {},{} for {}", wf.supported[i].first, wf.supported[i].second, wf.inferred_bases[i]);
            bases_map.insert({wf.supported[i], wf.inferred_bases[i]});
        }
        auto bases = wf.bases.squeeze(0);
        int tpos = -1, ins = 0;
        for (int c = 0; c < bases.sizes()[0]; c++) {
            const auto tbase = bases[c][0].item<uint8_t>();
            if (base_decoding[tbase] == '*') {
                ins += 1;
            } else {
                tpos += 1;
                ins = 0;
            }

            auto p = std::make_pair(tpos, ins);
            auto found_p = bases_map.find(p);
            if (found_p != bases_map.end()) {
                auto new_base = found_p->second;
                if (new_base != '*') {
                    corrected_seq += new_base;
                    //spdlog::info("{} tbase {} inferred base {} at {} {}", c, tbase, new_base, tpos, ins);
                }
            } else {
                std::array<base_count_t, 5> counter;
                for (int r = 0; r < wf.n_alns; r++) {
                    auto base = bases[c][r].item<uint8_t>();
                    if (base_decoding[base] == '.') {
                        continue;
                    }
                    auto idx = base_to_idx[base_decoding[base]];
                    counter[idx].b = base;
                    counter[idx].c++;
                }

                std::sort(counter.begin(), counter.end(),
                          [](base_count_t a, base_count_t b) { return a.c > b.c; });

                auto& first = counter[0];
                auto& second = counter[1];

                char new_base;
                if ((first.c < 2) ||
                    (first.c == second.c && (first.b == tbase || second.b == tbase))) {
                    new_base = base_decoding[tbase];
                } else {
                    new_base = base_decoding[first.b];
                }

                new_base = base_forward[new_base];
                if (new_base != '*') {
                    //spdlog::info("{} tbase {} new base {}", c, tbase, new_base);
                    corrected_seq += new_base;
                }
            }
        }
    }

    if (!corrected_seq.empty()) {
        spdlog::info("added seq end of len {}", corrected_seq.length());
        corrected_reads.push_back(corrected_seq);
    }

    return corrected_reads;
}

BamPtr create_bam_record(const std::string& read_id, const std::string& seq) {
    bam1_t* rec = bam_init1();
    bam_set1(rec, read_id.length(), read_id.c_str(), 4 /*flag*/, -1 /*tid*/, -1 /*pos*/, 0 /*mapq*/,
             0 /*n_cigar*/, nullptr /*cigar*/, -1 /*mtid*/, -1 /*mpos*/, 0 /*isize*/, seq.size(),
             seq.data(), nullptr, 0);
    return BamPtr(rec);
}

CorrectionNode::CorrectionNode(int threads) : MessageSink(10000, threads) {
    start_input_processing(&CorrectionNode::input_thread_fn, this);
}

void CorrectionNode::input_thread_fn() {
    Message message;
    mm_tbuf_t* tbuf = mm_tbuf_init();

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/home/OXFORDNANOLABS/jdaw/github/haec-BigBird/ont-model.pt");
    } catch (const c10::Error& e) {
        spdlog::error("Error loading model");
        throw std::runtime_error("");
    }

    spdlog::info("Loaded model!");

    while (get_input_message(message)) {
        if (std::holds_alternative<CorrectionAlignments>(message)) {
            auto alignments = std::get<CorrectionAlignments>(std::move(message));
            //if (alignments.read_name == "0db071fd-ac9e-47f2-ae42-de43c48abb43") {
            if (true) {
                spdlog::info("Process windows for {} of length", alignments.read_name,
                             alignments.read_seq.length());
                size_t n_windows =
                        (alignments.read_seq.length() + m_window_size - 1) / m_window_size;
                spdlog::info("num windows {}", n_windows);
                std::vector<std::vector<OverlapWindow>> windows;
                windows.resize(n_windows);
                extract_windows(windows, alignments, m_window_size);
                int o = 0;
                for (auto& ovlp_windows : windows) {
                    spdlog::info("{} ovlps in window {}", ovlp_windows.size(), o++);
                }
                auto wfs = extract_features(windows, alignments, m_window_size);
                run_inference(module, wfs);
                auto corrected_seqs = decode_windows(wfs);
                if (corrected_seqs.size() == 1) {
                    auto rec = create_bam_record(alignments.read_name, corrected_seqs[0]);
                    send_message_to_sink(std::move(rec));
                } else {
                    for (size_t s = 0; s < corrected_seqs.size(); s++) {
                        const std::string read_name =
                                alignments.read_name + ":" + std::to_string(s);
                        auto rec = create_bam_record(read_name, corrected_seqs[s]);
                        send_message_to_sink(std::move(rec));
                    }
                }
            }
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }
    mm_tbuf_destroy(tbuf);
}

stats::NamedStats CorrectionNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
