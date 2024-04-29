#include "features.h"

#include "conversions.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

namespace dorado::correction {

bool overlap_has_long_indel(const OverlapWindow& overlap, const CorrectionAlignments& alignments) {
    bool long_indel = false;
    const auto& cigar = alignments.cigars[overlap.overlap_idx];
    size_t max_cigar_idx = std::min(size_t(overlap.cigar_end_idx + 1), cigar.size());
    for (size_t i = overlap.cigar_start_idx; i < max_cigar_idx; i++) {
        if (cigar[i].op == CigarOpType::INS || cigar[i].op == CigarOpType::DEL) {
            long_indel |= cigar[i].len >= 30;
        }
    }
    spdlog::trace("filter ? tstart {} qstart {} qend {} res {}", overlap.tstart, overlap.qstart,
                  overlap.qend, long_indel);
    return long_indel;
}

// Measure the accuracy of an alignment segment within a window
// determined from the cigar string.
void calculate_accuracy(OverlapWindow& overlap,
                        const CorrectionAlignments& alignments,
                        size_t win_idx,
                        int win_len,
                        int window_size) {
    int tstart = overlap.tstart;
    int tend = (int)win_idx * window_size + win_len;

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

    spdlog::trace("tstart {} tend {} qstart {} qend {} cig st {} cig end {}", tstart, tend, qstart,
                  qend, overlap.cigar_start_idx, overlap.cigar_end_idx);

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

        spdlog::trace("len {} tpos {} qpos {}", len, tpos, qpos);

        switch (cigar[idx].op) {
        case CigarOpType::MATCH:
            for (int j = 0; j < len; j++) {
                auto tbase = tseq[tpos + j];
                auto qbase = qseq[qpos + j];
                spdlog::trace("{} tbase {}, {} qbase {}", tpos + j, tbase, qpos + j, qbase);

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

    overlap.accuracy = (static_cast<float>(m) / (m + s + i + d));
    spdlog::trace("m {} s {} i {} d {}", m, s, i, d);
    spdlog::trace("accuracy qstart {} qend {} {}", overlap.qstart, overlap.qend, overlap.accuracy);
}

// Calculate the maximum number of possible inserts for each position of the
// target sequence. This is done by looking at all aligned queries at each position
// and picking the longest insertion size.
std::vector<int> get_max_ins_for_window(const std::vector<OverlapWindow>& overlaps,
                                        const CorrectionAlignments& alignments,
                                        int tstart,
                                        int win_len) {
    std::vector<int> max_ins(win_len, 0);
    for (const auto& overlap : overlaps) {
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

    return max_ins;
}

// Generate the tensor encoding for each chunk/window. This function
// reads the bases from the target and query sequences and qualitiy
// scores and fills 2x2D matrices where each column in a position
// in the pileup and each row is a read.
std::tuple<torch::Tensor, torch::Tensor> get_features_for_window(
        const std::vector<OverlapWindow>& overlaps,
        const CorrectionAlignments& alignments,
        int win_len,
        int tstart,
        const std::vector<int>& max_ins) {
    static auto base_encoding = gen_base_encoding();
    static auto base_decoding = gen_base_decoding();
    auto bases_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto quals_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    const int length = std::accumulate(max_ins.begin(), max_ins.end(), 0) + (int)max_ins.size();
    const int reads = 1 + TOP_K;

    auto bases = torch::empty({reads, length}, bases_options);
    std::fill(bases.data_ptr<int>(), bases.data_ptr<int>() + bases.numel(), base_encoding['.']);
    auto quals = torch::empty({reads, length}, quals_options);
    std::fill(quals.data_ptr<float>(), quals.data_ptr<float>() + quals.numel(),
              normalize_quals((float)'!'));

    // Write bases/qual for target read
    const std::string& tseq = alignments.read_seq;
    const std::vector<uint8_t>& tqual = alignments.read_qual;

    int tpos = 0;
    int* target_bases_tensor = bases.data_ptr<int>();
    std::fill(target_bases_tensor, target_bases_tensor + length, base_encoding['*']);
    float* target_quals_tensor = quals.data_ptr<float>();
    // PyTorch stores data in column major format.
    for (int i = 0; i < win_len; i++) {
        target_bases_tensor[tpos] = base_encoding[tseq[i + tstart]];
        target_quals_tensor[tpos] = normalize_quals(float(tqual[i + tstart] + 33));

        spdlog::trace("tpos {} base {} qual {}", tpos, base_decoding[target_bases_tensor[tpos]],
                      target_quals_tensor[tpos]);
        tpos += 1 + max_ins[i];
    }

    // Write bases for each overlap in the window
    for (int w = 0; w < (int)overlaps.size(); w++) {
        spdlog::trace("get_features_for_ol_window for window {}", w);
        int* query_bases_tensor = &target_bases_tensor[length * (w + 1)];
        float* query_quals_tensor = &target_quals_tensor[length * (w + 1)];
        const auto& overlap = overlaps[w];
        const auto& cigar = alignments.cigars[overlap.overlap_idx];
        int offset = overlap.tstart - tstart;

        bool fwd = alignments.overlaps[overlap.overlap_idx].fwd;
        int oqstart = alignments.overlaps[overlap.overlap_idx].qstart;
        int oqend = alignments.overlaps[overlap.overlap_idx].qend;

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

        spdlog::trace(
                "qstart {} qend {} aln qstart {} aln qend {} overlap qstart {} overlap qend {}",
                qstart, qend, oqstart, oqend, overlap.qstart, overlap.qend);
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

        std::fill(query_bases_tensor, query_bases_tensor + length, base_encoding[gap]);

        tpos = offset;
        int idx = offset + std::accumulate(max_ins.begin(), max_ins.begin() + offset, 0);

        spdlog::trace("cigar_len {}, cigar_end {}, gap {}, tpos {}, idx {}, fwd {}", cigar_len,
                      cigar_end, gap, tpos, idx, fwd ? '+' : '-');

        if (idx > 0) {
            std::fill(query_bases_tensor, query_bases_tensor + idx, base_encoding['.']);
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

            spdlog::trace("cigar_idx {} l {}", cigar_idx, l);

            switch (op) {
            case CigarOpType::MATCH:
            case CigarOpType::MISMATCH:
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    query_bases_tensor[idx] = base;
                    query_quals_tensor[idx] = normalize_quals((float)qual + 33);

                    spdlog::trace("idx {} base {}, qual {}", idx,
                                  base_decoding[query_bases_tensor[idx]], query_quals_tensor[idx]);

                    idx += 1 + max_ins[tpos + i];
                    query_iter++;
                }

                tpos += l;
                break;
            case CigarOpType::DEL:
                for (uint32_t i = 0; i < l; i++) {
                    spdlog::trace("idx {}", idx);
                    idx += 1 + max_ins[tpos + i];
                }
                tpos += l;
                break;
            case CigarOpType::INS:
                idx -= max_ins[tpos - 1];
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    query_bases_tensor[(idx + i)] = base;
                    query_quals_tensor[(idx + i)] = normalize_quals((float)qual + 33);

                    spdlog::trace("idx + i {} base {}, qual {}", idx + i,
                                  base_decoding[query_bases_tensor[(idx + i)]],
                                  query_quals_tensor[(idx + i)]);

                    query_iter++;
                }

                idx += max_ins[tpos - 1];
            }
        }

        if (idx < length) {
            std::fill(query_bases_tensor + idx, query_bases_tensor + length, base_encoding['.']);
        }

        spdlog::trace("sum of bases at at overlap {} {}", w, bases.sum().item<int>());
    }

    return {std::move(bases), std::move(quals)};
}

// From the encoding get positions of the target
// read that are candidates for the network to correct.
// A position qualifies as a candidate if majority voting
// in that column surfaces at least 2 different bases
// each with a minimum of 3 supporting reads.
// The candidate positions are returned as a tuple with the
// first element being a position in the target sequence
// and the second element being an insertion offset from that
// position.
std::vector<std::pair<int, int>> get_supported(torch::Tensor& bases) {
    std::vector<std::pair<int, int>> supported;

    static auto base_forward = base_forward_mapping();
    static auto base_encoding = gen_base_encoding();
    static auto base_decoding = gen_base_decoding();

    const int reads = bases.sizes()[0];
    const int length = bases.sizes()[1];

    auto bases_ptr = bases.data_ptr<int>();

    int tpos = -1, ins = 0;
    std::array<int, 128> counter;
    for (int c = 0; c < length; c++) {
        if (bases_ptr[c] == base_encoding['*']) {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }
        counter.fill(0);
        for (int r = 0; r < reads; r++) {
            auto base = bases_ptr[r * length + c];
            spdlog::trace("row {} base {}", r, base);
            if (base == base_encoding['.']) {
                continue;
            }
            counter[base_forward[base_decoding[base]]]++;
        }

        spdlog::trace("col {} A {} C {} T {} G {} * {}", c, counter['A'], counter['C'],
                      counter['T'], counter['G'], counter['*']);
        int count = std::count_if(counter.begin(), counter.end(), [](int num) { return num >= 3; });
        if (count >= 2) {
            supported.push_back({tpos, ins});
            spdlog::trace("support added for {} {}", tpos, ins);
        }
        spdlog::trace("num supported {}", supported.size());
    }
    return supported;
}

// Convert the tuple of pairs for {target pos, insertion offset} into a
// column in the tensor.
torch::Tensor get_indices(const torch::Tensor& bases,
                          const std::vector<std::pair<int, int>>& supported) {
    static auto base_encoding = gen_base_encoding();
    auto tbase_tensor = bases.data_ptr<int>();
    std::vector<int> indices;
    for (int i = 0; i < bases.sizes()[1]; i++) {
        if (tbase_tensor[i] != base_encoding['*']) {
            indices.push_back(i);
        }
    }

    std::vector<int> supported_indices;
    supported_indices.reserve(supported.size());
    for (auto [pos, ins] : supported) {
        supported_indices.push_back(indices[pos] + ins);
    }

    return torch::from_blob(supported_indices.data(), {(int)supported_indices.size()},
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
            .clone();
}

// Main interface function for generating features for each window
// given the overlaps for a target read.
std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments,
                                             int window_size) {
    const std::string& tseq = alignments.read_seq;
    int tlen = tseq.length();

    std::vector<WindowFeatures> wfs;
    for (size_t w = 0; w < windows.size(); w++) {
        int win_len = (w == windows.size() - 1) ? tlen - window_size * w : window_size;
        spdlog::trace("win idx {}: win len {}", w, win_len);
        auto& overlap_windows = windows[w];

        // Filter overlaps with very large indels
        std::vector<OverlapWindow> filtered_overlaps;
        for (auto& ovlp : overlap_windows) {
            if (!overlap_has_long_indel(ovlp, alignments)) {
                filtered_overlaps.push_back(std::move(ovlp));
            }
        }
        spdlog::trace("window {} pre filter windows {} post filter windows {}", w,
                      overlap_windows.size(), filtered_overlaps.size());
        overlap_windows = std::move(filtered_overlaps);
        //windows[w] = std::move(filtered_overlaps);

        // Sort overlaps by score
        if (overlap_windows.size() > 1) {
            for (auto& ovlp : overlap_windows) {
                calculate_accuracy(ovlp, alignments, w, win_len, window_size);
            }
            // Sort the filtered overlaps by accuracy score
            std::sort(overlap_windows.begin(), overlap_windows.end(),
                      [](const OverlapWindow& a, const OverlapWindow& b) {
                          return a.accuracy > b.accuracy;
                      });
        }
        overlap_windows.resize(std::min(TOP_K, (int)overlap_windows.size()));

        if (overlap_windows.size() == 1) {
            spdlog::trace("window {} 1st {}-{}", w, overlap_windows[0].qstart,
                          overlap_windows[0].qend);
        } else if (overlap_windows.size() > 1) {
            spdlog::trace("window {} 1st {}-{} 2nd {}-{}", w, overlap_windows[0].qstart,
                          overlap_windows[0].qend, overlap_windows[1].qstart,
                          overlap_windows[1].qend);
        }

        WindowFeatures wf;
        wf.window_idx = w;
        wf.read_name = alignments.read_name;
        wf.n_alns = (int)overlap_windows.size();
        if (overlap_windows.size() > 1) {
            // Find the maximum insert size
            auto max_ins =
                    get_max_ins_for_window(overlap_windows, alignments, w * window_size, win_len);

            // Create tensors
            auto [bases, quals] = get_features_for_window(overlap_windows, alignments, win_len,
                                                          w * window_size, max_ins);
            auto supported = get_supported(bases);
            wf.bases = std::move(bases);
            wf.quals = std::move(quals);
            wf.supported = std::move(supported);
            wf.length = (int)wf.supported.size();
            wf.indices = get_indices(wf.bases, wf.supported);
        }
        wfs.push_back(std::move(wf));
    }

    return wfs;
}

}  // namespace dorado::correction
