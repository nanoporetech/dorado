#include "features.h"

#include "conversions.h"
#include "correct/types.h"
#include "read_pipeline/messages.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/cigar.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <ATen/Tensor.h>
#include <spdlog/spdlog.h>
#include <torch/types.h>

#include <cstdint>
#include <stdexcept>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

constexpr int32_t TOP_K = 30;
constexpr int32_t MAX_INDEL_LEN = 30;

namespace dorado::correction {

namespace {

bool overlap_has_long_indel(const OverlapWindow& overlap,
                            const CorrectionAlignments& alignments,
                            const int32_t max_indel_len) {
    const auto& cigar = alignments.cigars[overlap.overlap_idx];
    const size_t max_cigar_idx = std::min(size_t(overlap.cigar_end_idx + 1), cigar.size());
    for (size_t i = overlap.cigar_start_idx; i < max_cigar_idx; i++) {
        if ((static_cast<int32_t>(cigar[i].len) >= max_indel_len) &&
            ((cigar[i].op == CigarOpType::I) || (cigar[i].op == CigarOpType::D))) {
            LOG_TRACE("filter ? tstart {} qstart {} qend {} long indel: {}", overlap.tstart,
                      overlap.qstart, overlap.qend, cigar[i].len);
            return true;
        }
    }
    return false;
}

// Measure the accuracy of an alignment segment within a window
// determined from the cigar string.
void calculate_accuracy(OverlapWindow& overlap, const CorrectionAlignments& alignments) {
    const auto& cigar = alignments.cigars[overlap.overlap_idx];
    bool has_warned_bad_cigar_op = false;

    // counts of match, mismatch, insert, deletion
    int n_match = 0, n_miss = 0, n_ins = 0, n_del = 0;

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

        switch (cigar[idx].op) {
        case CigarOpType::EQ:
            n_match += len;
            break;
        case CigarOpType::X:
            n_miss += len;
            break;
        case CigarOpType::I:
            n_ins += len;
            break;
        case CigarOpType::D:
            n_del += len;
            break;
        default:
            if (!has_warned_bad_cigar_op) {
                has_warned_bad_cigar_op = true;
                LOG_TRACE("unexpected CigarOpType: {}", static_cast<uint8_t>(cigar[idx].op));
            }
            break;
        }
    }

    overlap.accuracy = (static_cast<float>(n_match) / (n_match + n_miss + n_ins + n_del));
    LOG_TRACE("m {} s {} i {} d {}", n_match, n_miss, n_ins, n_del);
    LOG_TRACE("accuracy qstart {} qend {} {}", overlap.qstart, overlap.qend, overlap.accuracy);
}

// Calculate the maximum number of possible inserts for each position of the
// target sequence. This is done by looking at all aligned queries at each position
// and picking the longest insertion size.
std::vector<int32_t> get_max_ins_for_window(const std::vector<OverlapWindow>& overlaps,
                                            const CorrectionAlignments& alignments) {
    if (std::empty(overlaps)) {
        return {};
    }
    const int32_t win_len = overlaps.front().win_tend - overlaps.front().win_tstart;
    std::vector<int> max_ins(win_len, 0);
    for (const auto& overlap : overlaps) {
        int32_t tpos = overlap.tstart - overlap.win_tstart;

        const auto& cigar = alignments.cigars[overlap.overlap_idx];
        const int32_t cigar_len = overlap.cigar_end_idx - overlap.cigar_start_idx + 1;

        for (int32_t i = overlap.cigar_start_idx;
             i <= std::min(overlap.cigar_end_idx, static_cast<int32_t>(std::size(cigar)) - 1);
             ++i) {
            const CigarOpType op = cigar[i].op;
            const int32_t len = cigar[i].len;

            int32_t l = 0;
            if ((op == CigarOpType::EQ) || (op == CigarOpType::X) || (op == CigarOpType::D)) {
                l = len;
            } else if ((op == CigarOpType::I) && (tpos > 0)) {
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
std::tuple<at::Tensor, at::Tensor> get_features_for_window(
        const std::vector<OverlapWindow>& overlaps,
        const CorrectionAlignments& alignments,
        const std::vector<int32_t>& max_ins) {
    if (std::empty(overlaps)) {
        return {};
    }
    const int32_t win_len = overlaps.front().win_tend - overlaps.front().win_tstart;
    const int32_t win_tstart = overlaps.front().win_tstart;

    static auto base_encoding = gen_base_encoding();
#ifndef NDEBUG
    static auto base_decoding = gen_base_decoding();
#endif
    auto bases_options = at::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto quals_options = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    const int length = std::accumulate(max_ins.begin(), max_ins.end(), 0) + (int)max_ins.size();
    const int reads = 1 + TOP_K;

    auto bases = at::empty({reads, length}, bases_options);
    std::fill(bases.data_ptr<int>(), bases.data_ptr<int>() + bases.numel(), base_encoding['.']);
    auto quals = at::empty({reads, length}, quals_options);
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
        target_bases_tensor[tpos] = base_encoding[tseq[i + win_tstart]];
        target_quals_tensor[tpos] = normalize_quals(float(tqual[i + win_tstart] + 33));

        LOG_TRACE("tpos {} base {} qual {}", tpos, base_decoding[target_bases_tensor[tpos]],
                  target_quals_tensor[tpos]);
        tpos += 1 + max_ins[i];
    }

    // Write bases for each overlap in the window
    for (int w = 0; w < (int)overlaps.size(); w++) {
        LOG_TRACE("get_features_for_ol_window for window {}", w);
        int* query_bases_tensor = &target_bases_tensor[length * (w + 1)];
        float* query_quals_tensor = &target_quals_tensor[length * (w + 1)];
        const auto& overlap = overlaps[w];
        const auto& cigar = alignments.cigars[overlap.overlap_idx];
        int offset = overlap.tstart - win_tstart;

        bool fwd = alignments.overlaps[overlap.overlap_idx].fwd;
        int oqstart = alignments.overlaps[overlap.overlap_idx].qstart;
        int oqend = alignments.overlaps[overlap.overlap_idx].qend;
        const int oqlen = alignments.overlaps[overlap.overlap_idx].qlen;

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

        LOG_TRACE("qstart {} qend {} aln qstart {} aln qend {} overlap qstart {} overlap qend {}",
                  qstart, qend, oqstart, oqend, overlap.qstart, overlap.qend);
        int query_iter = 0;
        if (qstart >= oqlen) {
            throw std::runtime_error{
                    "Query start coordinate is out of bounds when computing features. Make sure "
                    "that the initial query start position in the window is set to 0, instead of "
                    "overlap qstart. qstart = " +
                    std::to_string(qstart) + ", oqlen = " + std::to_string(oqlen)};
        }
        std::string qseq = alignments.seqs[overlap.overlap_idx].substr(qstart, qlen);
        std::vector<uint8_t> qqual(alignments.quals[overlap.overlap_idx].begin() + qstart,
                                   alignments.quals[overlap.overlap_idx].begin() + qend);
        if (!fwd) {
            qseq = utils::reverse_complement(qseq);
            std::reverse(qqual.begin(), qqual.end());
        }

        const int cigar_len_total = static_cast<int>(std::size(cigar));
        const int cigar_len = overlap.cigar_end_idx - overlap.cigar_start_idx + 1;
        const int cigar_end = std::min(cigar_len_total - overlap.cigar_start_idx, cigar_len);

        uint8_t gap = fwd ? '*' : '#';

        std::fill(query_bases_tensor, query_bases_tensor + length, base_encoding[gap]);

        tpos = offset;
        int idx = offset + std::accumulate(max_ins.begin(), max_ins.begin() + offset, 0);

        LOG_TRACE("cigar_len {}, cigar_end {}, gap {}, tpos {}, idx {}, fwd {}", cigar_len,
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

            LOG_TRACE("cigar_idx {} l {}", cigar_idx, l);

            switch (op) {
            case CigarOpType::EQ:
            case CigarOpType::X:
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    query_bases_tensor[idx] = base;
                    query_quals_tensor[idx] = normalize_quals((float)(qual + 33));

                    LOG_TRACE("idx {} base {}, qual {}", idx,
                              base_decoding[query_bases_tensor[idx]], query_quals_tensor[idx]);

                    idx += 1 + max_ins[tpos + i];
                    query_iter++;
                }

                tpos += l;
                break;
            case CigarOpType::D:
                for (uint32_t i = 0; i < l; i++) {
                    LOG_TRACE("idx {}", idx);
                    idx += 1 + max_ins[tpos + i];
                }
                tpos += l;
                break;
            case CigarOpType::I:
                idx -= max_ins[tpos - 1];
                for (uint32_t i = 0; i < l; i++) {
                    auto base = base_encoding[uint8_t(qseq[query_iter]) + (fwd ? 0 : 32)];
                    auto qual = qqual[query_iter];

                    query_bases_tensor[(idx + i)] = base;
                    query_quals_tensor[(idx + i)] = normalize_quals((float)(qual + 33));

                    LOG_TRACE("idx + i {} base {}, qual {}", idx + i,
                              base_decoding[query_bases_tensor[(idx + i)]],
                              query_quals_tensor[(idx + i)]);

                    query_iter++;
                }

                idx += max_ins[tpos - 1];
                break;
            default:
                throw std::runtime_error(
                        "Unsupported CIGAR operation found in get_features_for_window! Op: " +
                        std::string(1, convert_cigar_op_to_char(op)));
                break;
            }
        }

        if (idx < length) {
            std::fill(query_bases_tensor + idx, query_bases_tensor + length, base_encoding['.']);
        }

        LOG_TRACE("sum of bases at at overlap {} {}", w, bases.sum().item<int>());
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
std::vector<std::pair<int, int>> get_supported(at::Tensor& bases) {
    std::vector<std::pair<int, int>> supported;

    static auto base_forward = base_forward_mapping();
    static auto base_encoding = gen_base_encoding();
    static auto base_decoding = gen_base_decoding();

    const int reads = static_cast<int>(bases.sizes()[0]);
    const int length = static_cast<int>(bases.sizes()[1]);

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
            LOG_TRACE("row {} base {}", r, base);
            if (base == base_encoding['.']) {
                continue;
            }
            counter[base_forward[base_decoding[base]]]++;
        }

        LOG_TRACE("col {} A {} C {} T {} G {} * {}", c, counter['A'], counter['C'], counter['T'],
                  counter['G'], counter['*']);
        int count = (int)std::count_if(counter.begin(), counter.end(),
                                       [](int num) { return num >= 3; });
        if (count >= 2) {
            supported.push_back({tpos, ins});
            LOG_TRACE("support added for {} {}", tpos, ins);
        }
        LOG_TRACE("num supported {}", supported.size());
    }
    return supported;
}

// Convert the tuple of pairs for {target pos, insertion offset} into a
// column in the tensor.
at::Tensor get_indices(const at::Tensor& bases, const std::vector<std::pair<int, int>>& supported) {
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

    return at::from_blob(supported_indices.data(), {(int)supported_indices.size()},
                         at::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
            .clone();
}

}  // namespace

std::unordered_set<int> filter_features(std::vector<std::vector<OverlapWindow>>& windows,
                                        const CorrectionAlignments& alignments) {
    utils::ScopedProfileRange spr("filter_features", 1);

    std::unordered_set<int> overlap_idxs;
    for (int w = 0; w < (int)windows.size(); w++) {
        auto& overlap_windows = windows[w];

        // Filter overlaps with very large indels
        std::vector<OverlapWindow> filtered_overlaps;
        for (auto& ovlp : overlap_windows) {
            if (!overlap_has_long_indel(ovlp, alignments, MAX_INDEL_LEN)) {
                filtered_overlaps.push_back(std::move(ovlp));
            }
        }
        LOG_TRACE("window {} pre filter windows {} post filter windows {}", w,
                  overlap_windows.size(), filtered_overlaps.size());
        overlap_windows = std::move(filtered_overlaps);

        // Sort overlaps by score
        if (overlap_windows.size() > 1) {
            for (auto& ovlp : overlap_windows) {
                calculate_accuracy(ovlp, alignments);
            }
            // Sort the filtered overlaps by accuracy score
            std::sort(overlap_windows.begin(), overlap_windows.end(),
                      [&alignments](const OverlapWindow& a, const OverlapWindow& b) {
                          if (std::fabs(a.accuracy - b.accuracy) < 1e-10) {
                              const auto& a_qname = alignments.qnames[a.overlap_idx];
                              const auto& b_qname = alignments.qnames[b.overlap_idx];
                              return std::lexicographical_compare(a_qname.begin(), a_qname.end(),
                                                                  b_qname.begin(), b_qname.end());
                          }
                          return a.accuracy > b.accuracy;
                      });
        }
        // Take the TOP_K best overlaps
        overlap_windows.resize(std::min(TOP_K, (int)overlap_windows.size()));

#ifndef NDEBUG
        if (overlap_windows.size() == 1) {
            LOG_TRACE("window {} 1st {}-{}", w, overlap_windows[0].qstart, overlap_windows[0].qend);
        } else if (overlap_windows.size() > 1) {
            LOG_TRACE("window {} 1st {}-{} 2nd {}-{}", w, overlap_windows[0].qstart,
                      overlap_windows[0].qend, overlap_windows[1].qstart, overlap_windows[1].qend);
        }
#endif

        for (const auto& ov : overlap_windows) {
            assert(ov.overlap_idx >= 0);
            overlap_idxs.insert(ov.overlap_idx);
        }
    }
    return overlap_idxs;
}

// Main interface function for generating features for the top_k overlaps for each window
// given the overlaps for a target read.
std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                             const CorrectionAlignments& alignments) {
    std::vector<WindowFeatures> wfs;
    for (int w = 0; w < (int)windows.size(); w++) {
        LOG_TRACE("win idx {}", w);
        auto& overlap_windows = windows[w];

        WindowFeatures wf;
        wf.window_idx = w;
        wf.read_name = alignments.read_name;
        wf.n_alns = (int)overlap_windows.size();
        if (overlap_windows.size() > 1) {
            // Find the maximum insert size
            const std::vector<int32_t> max_ins =
                    get_max_ins_for_window(overlap_windows, alignments);

            // Create tensors
            auto [bases, quals] = get_features_for_window(overlap_windows, alignments, max_ins);
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
