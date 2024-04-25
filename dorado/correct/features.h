#pragma once

#include "types.h"

#include <spdlog/spdlog.h>

namespace dorado::correction {

const int TOP_K = 30;

bool filter_overlap(const OverlapWindow& overlap, const CorrectionAlignments& alignments) {
    bool long_indel = false;
    const auto& cigar = alignments.cigars[overlap.overlap_idx];
    for (size_t i = overlap.cigar_start_idx;
         i < std::min(size_t(overlap.cigar_end_idx + 1), cigar.size()); i++) {
        if (cigar[i].op == CigarOpType::INS || cigar[i].op == CigarOpType::DEL) {
            long_indel |= cigar[i].len >= 30;
        }
    }
    //spdlog::info("filter ? tstart {} qstart {} qend {} res {}", overlap.tstart, overlap.qstart, overlap.qend, long_indel);
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

    bool print = false;
    if (overlap.qstart == 16347 && overlap.qend == 20443) {
        print = true;
        //spdlog::info("cigar start idx {} cigar end idx {}", overlap.cigar_start_idx, overlap.cigar_end_idx);
    }

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

        if (print) {
            //spdlog::info("len {} tpos {} qpos {}", len, tpos, qpos);
        }

        switch (cigar[idx].op) {
        case CigarOpType::MATCH:
            for (int j = 0; j < len; j++) {
                auto tbase = tseq[tpos + j];
                auto qbase = qseq[qpos + j];
                if (print) {
                    //spdlog::info("{} tbase {}, {} qbase {}", tpos + j, tbase, qpos + j, qbase);
                }

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
    //spdlog::info("accuracy qstart {} qend {} {}", overlap.qstart, overlap.qend, overlap.accuracy);
}

std::vector<int> get_max_ins_for_window(const std::vector<OverlapWindow>& overlaps,
                                        const CorrectionAlignments& alignments,
                                        int tstart,
                                        int win_len) {
    std::vector<int> max_ins(win_len, 0);
    ;
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

    //int sum = 0;
    //for (auto i : max_ins) {
    //    sum += i;
    //}
    //spdlog::info("max ins sum {}", sum);
    return max_ins;
}

std::tuple<torch::Tensor, torch::Tensor> get_features_for_window(
        const std::vector<OverlapWindow>& overlaps,
        const CorrectionAlignments& alignments,
        int win_len,
        int tstart,
        const std::vector<int>& max_ins) {
    std::chrono::duration<double> string_time{};
    auto t_init = std::chrono::high_resolution_clock::now();
    static auto base_encoding = gen_base_encoding();
    auto bases_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto quals_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    int length = std::accumulate(max_ins.begin(), max_ins.end(), 0) + (int)max_ins.size();
    int reads = 1 + TOP_K;
    //int* alloc_bases_ptr = m_bases_manager.get_next_ptr();
    //float* alloc_quals_ptr = m_quals_manager.get_next_ptr();

    //auto bases = torch::from_blob(alloc_bases_ptr, {reads, length}, bases_options);
    //auto quals = torch::from_blob(alloc_quals_ptr, {reads, length}, quals_options);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = (t0 - t_init);

    auto bases = torch::empty({reads, length}, bases_options);
    std::fill(bases.data_ptr<int>(), bases.data_ptr<int>() + bases.numel(), base_encoding['.']);
    //bases.fill_(base_encoding['.']);
    //auto bases = torch::full({reads, length}, base_encoding['.'], bases_options);
    auto quals = torch::empty({reads, length}, quals_options);
    std::fill(quals.data_ptr<float>(), quals.data_ptr<float>() + quals.numel(),
              normalize_quals((float)'!'));
    //quals.fill_((float)'!');
    //auto quals = torch::full({reads, length}, (float)'!', quals_options);

    auto tfill = std::chrono::high_resolution_clock::now();
    dur = (tfill - t0);

    //spdlog::info("time to create tensor {}", dur.count());
    // Write target for window
    const std::string& tseq = alignments.read_seq;
    const std::vector<uint8_t>& tqual = alignments.read_qual;

    int tpos = 0;
    //auto target_bases_tensor = bases.index({torch::indexing::Slice(), 0});
    int* target_bases_tensor = bases.data_ptr<int>();
    //target_bases_tensor.fill_(base_encoding['*']);
    std::fill(target_bases_tensor, target_bases_tensor + length, base_encoding['*']);
    //auto target_quals_tensor = quals.index({torch::indexing::Slice(), 0});
    float* target_quals_tensor = quals.data_ptr<float>();
    //for(int f = 0; f < length; f++) {
    //    spdlog::info("qual default {}", target_quals_tensor[f]);
    //}
    // PyTorch stores data in column major format.
    for (int i = 0; i < win_len; i++) {
        target_bases_tensor[tpos] = base_encoding[tseq[i + tstart]];
        target_quals_tensor[tpos] = normalize_quals(float(tqual[i + tstart] + 33));

        //spdlog::info("tpos {} base {} qual {}", tpos, base_decoding[target_bases_tensor[tpos]], target_quals_tensor[tpos]);
        tpos += 1 + max_ins[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t1 - t0;
    //spdlog::info("prolog duration {}", duration.count());
    string_time += duration;

    //std::exit(0);

    // Write bases for each overlap window
    for (int w = 0; w < (int)overlaps.size(); w++) {
        t0 = std::chrono::high_resolution_clock::now();
        //spdlog::info("get_features_for_ol_window for window {}", w);
        //auto query_bases_tensor = bases.index({torch::indexing::Slice(), w + 1});
        int* query_bases_tensor =
                &target_bases_tensor[length *
                                     (w + 1)];  //bases.index({torch::indexing::Slice(), w + 1});
        //auto query_quals_tensor = quals.index({torch::indexing::Slice(), w + 1});
        float* query_quals_tensor =
                &target_quals_tensor[length *
                                     (w + 1)];  //quals.index({torch::indexing::Slice(), w + 1});
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

        //spdlog::info("qstart {} qend {} aln qstart {} aln qend {} overlap qstart {} overlap qend {}", qstart, qend, oqstart, oqend, overlap.qstart, overlap.qend);
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

        //query_bases_tensor.fill_(base_encoding[gap]);
        std::fill(query_bases_tensor, query_bases_tensor + length, base_encoding[gap]);

        tpos = offset;
        int idx = offset + std::accumulate(max_ins.begin(), max_ins.begin() + offset, 0);

        //spdlog::info("cigar_len {}, cigar_end {}, gap {}, tpos {}, idx {}, fwd {}", cigar_len, cigar_end, gap, tpos, idx, fwd ? '+' : '-');

        if (idx > 0) {
            //auto no_alignment = query_bases_tensor.index({torch::indexing::Slice(0, idx)});
            //no_alignment.fill_(base_encoding['.']);
            std::fill(query_bases_tensor, query_bases_tensor + idx, base_encoding['.']);
            //spdlog::info("fill idx {} with {}", p, base_decoding[query_bases_tensor[p]]);
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

                    query_bases_tensor[idx] = base;
                    query_quals_tensor[idx] = normalize_quals((float)qual + 33);

                    //spdlog::info("idx {} base {}, qual {}", idx, base_decoding[query_bases_tensor[idx]], query_quals_tensor[idx]);

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

                    query_bases_tensor[(idx + i)] = base;
                    query_quals_tensor[(idx + i)] = normalize_quals((float)qual + 33);

                    //spdlog::info("idx + i {} base {}, qual {}", idx + i, base_decoding[query_bases_tensor[(idx + i)]], query_quals_tensor[(idx + i)]);

                    query_iter++;
                }

                idx += max_ins[tpos - 1];
            }
        }

        //if (idx < query_bases_tensor.sizes()[0]) {
        if (idx < length) {
            //auto no_alignment =
            //        query_bases_tensor.index({torch::indexing::Slice(idx, torch::indexing::None)});
            //no_alignment.fill_(base_encoding['.']);
            std::fill(query_bases_tensor + idx, query_bases_tensor + length, base_encoding['.']);
        }

        //std::exit(0);
        //spdlog::info("sum of bases at at overlap {} {}",w , bases.sum().item<int>());
        t1 = std::chrono::high_resolution_clock::now();
        duration = t1 - t0;
        //spdlog::info("duration for overlap {}: {}", w, duration.count());
        string_time += duration;
    }

    //for(int i = 0; i < length; i++) {
    //    spdlog::info("target row at end pos {} base {}", i, bases[0][1].item<int>());
    //}

    //spdlog::info("string time {}", string_time.count());
    return {std::move(bases), std::move(quals)};
}

std::vector<std::pair<int, int>> get_supported(torch::Tensor& bases) {
    std::vector<std::pair<int, int>> supported;

    static auto base_forward = base_forward_mapping();
    static auto base_encoding = gen_base_encoding();
    static auto base_decoding = gen_base_decoding();

    const int reads = bases.sizes()[0];
    const int length = bases.sizes()[1];

    //spdlog::info("cols {} rows {}", cols, rows);

    auto bases_ptr = bases.data_ptr<int>();

    int tpos = -1, ins = 0;
    std::array<int, 128> counter;
    for (int c = 0; c < length; c++) {
        //if (bases[c][0].item<uint8_t>() == base_encoding['*']) {
        if (bases_ptr[c] == base_encoding['*']) {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }
        counter.fill(0);
        for (int r = 0; r < reads; r++) {
            //auto base = bases[c][r].item<uint8_t>();
            auto base = bases_ptr[r * length + c];
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

torch::Tensor get_indices(torch::Tensor bases, std::vector<std::pair<int, int>>& supported) {
    static auto base_encoding = gen_base_encoding();
    //auto tbase_tensor = bases.index({torch::indexing::Slice(), 0});
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

        auto t0 = std::chrono::high_resolution_clock::now();
        // Filter overlaps with very large indels
        std::vector<OverlapWindow> filtered_overlaps;
        for (auto& ovlp : overlap_windows) {
            if (!filter_overlap(ovlp, alignments)) {
                filtered_overlaps.push_back(std::move(ovlp));
            }
        }
        //spdlog::info("window {} pre filter windows {} post filter windows {}", w,
        //             overlap_windows.size(), filtered_overlaps.size());
        windows[w] = std::move(filtered_overlaps);

        auto t1 = std::chrono::high_resolution_clock::now();
        //for (auto& ovlp : windows[w]) {
        //    spdlog::info("qstart {} qend {}", ovlp.qstart, ovlp.qend);
        //}

        // Sort overlaps by score
        if (windows[w].size() > 1) {
            for (auto& ovlp : windows[w]) {
                calculate_accuracy(ovlp, alignments, w, win_len, m_window_size);
            }
            // Sort the filtered overlaps by accuracy score
            std::sort(windows[w].begin(), windows[w].end(),
                      [](const OverlapWindow& a, const OverlapWindow& b) {
                          return a.accuracy > b.accuracy;
                      });
        }
        windows[w].resize(std::min(TOP_K, (int)windows[w].size()));
        auto t2 = std::chrono::high_resolution_clock::now();

        if (windows[w].size() == 1) {
            //spdlog::info("window {} 1st {}-{}", w, windows[w][0].qstart, windows[w][0].qend);
        }
        if (windows[w].size() > 1) {
            //spdlog::info("window {} 1st {}-{} 2nd {}-{}", w, windows[w][0].qstart, windows[w][0].qend, windows[w][1].qstart, windows[w][1].qend);
        }

        WindowFeatures wf;
        wf.window_idx = w;
        wf.read_name = alignments.read_name;
        wf.n_alns = (int)windows[w].size();
        if (windows[w].size() > 1) {
            // Find the maximum insert size
            auto start = std::chrono::high_resolution_clock::now();
            auto max_ins =
                    get_max_ins_for_window(windows[w], alignments, w * m_window_size, win_len);
            auto ins = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = ins - start;

            // Create tensors
            //spdlog::info("get features for window {}", w);
            auto [bases, quals] = get_features_for_window(windows[w], alignments, win_len,
                                                          w * m_window_size, max_ins);
            auto feat = std::chrono::high_resolution_clock::now();
            duration = feat - ins;
            //spdlog::info("time to get features for window {} is {}", w, duration.count());
            auto supported = get_supported(bases);
            //spdlog::info("num supported {}", supported.size());
            auto supp = std::chrono::high_resolution_clock::now();

            duration = supp - feat;

            wf.bases = std::move(bases);
            wf.quals = std::move(quals);
            wf.supported = std::move(supported);
            wf.length = (int)wf.supported.size();
            //    std::move(
            //        torch::full({1}, (int)wf.supported.size(),
            //                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)));
            wf.indices = get_indices(wf.bases, wf.supported);
            auto ind = std::chrono::high_resolution_clock::now();

            duration = ind - supp;
        }
        wfs.push_back(std::move(wf));
        auto t3 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = t1 - t0;
        duration = t2 - t1;
        duration = t3 - t2;
    }

    return wfs;
}

}  // namespace dorado::correction
