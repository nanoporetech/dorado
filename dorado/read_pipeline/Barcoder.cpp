#include "Barcoder.h"

#include "htslib/sam.h"
#include "minimap.h"
//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include "3rdparty/edlib/edlib/include/edlib.h"
#include "mmpriv.h"
#include "utils/alignment_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

const std::string UNCLASSIFIED_BARCODE = "unclassified";

Barcoder::Barcoder(MessageSink& sink,
                   const std::vector<std::string>& barcodes,
                   int threads,
                   int k,
                   int w,
                   int m,
                   int q,
                   const std::string& barcode_file,
                   const std::string& kit_name)
        : MessageSink(10000), m_sink(sink), m_threads(threads), m_q(q), m_kit_name(kit_name) {
    read_barcodes(barcode_file);

    std::cerr << "FILE " << barcode_file << std::endl;
    init_mm2_settings(k, w, m);

    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&Barcoder::worker_thread, this, i)));
    }
}

Barcoder::~Barcoder() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_destroy(m_index);
    // Adding for thread safety in case worker thread throws exception.
    m_sink.terminate();
}

void Barcoder::init_mm2_settings(int k, int w, int m) {
    // Initialization for minimap2 based barcoding.
    // Initialize option structs.
    mm_set_opt(0, &m_idx_opt, &m_map_opt);
    // Setting options to map-ont default till relevant args are exposed.
    m_idx_opt.k = k;
    m_idx_opt.w = w;

    m_map_opt.min_chain_score = m;
    m_map_opt.min_cnt = 5;
    m_map_opt.occ_dist = 5;
    //m_map_opt.mid_occ_frac = 0.9;

    mm_check_opt(&m_idx_opt, &m_map_opt);

    int num_barcodes = m_barcodes.size();
    auto seqs = std::make_unique<const char*[]>(num_barcodes);
    auto names = std::make_unique<const char*[]>(num_barcodes);
    int index = 0;
    for (auto& [bc, seq] : m_barcodes) {
        seqs[index] = seq.c_str();
        names[index] = bc.c_str();
        spdlog::info("{}: {} - length {}", bc, seq, seq.length());
        index++;
    }

    m_index = mm_idx_str(m_idx_opt.w, m_idx_opt.k, 0, m_idx_opt.bucket_bits, num_barcodes,
                         seqs.get(), names.get());

    if (true /*mm_verbose >= 3*/) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }
}

void Barcoder::worker_thread(size_t tid) {
    m_active++;  // Track active threads.

    Message message;
    while (m_work_queue.try_pop(message)) {
        auto read = std::get<BamPtr>(std::move(message));
        auto records = barcode(read.get(), m_tbufs[tid]);
        for (auto& record : records) {
            m_sink.push_message(std::move(record));
        }
    }

    int num_active = --m_active;
    if (num_active == 0) {
        terminate();
        m_sink.terminate();
        spdlog::info("> Barcoded {}", m_matched.load());
    }
}

int calculate_gaps(const EdlibAlignResult& res) {
    int count = 0;
    for (int i = 0; i < res.alignmentLength; i++) {
        if (res.alignment[i] == 2 || res.alignment[i] == 3)
            count++;
    }
    return count;
}

int calculate_edit_dist(const EdlibAlignResult& res, int flank_len, int query_len) {
    int dist = 0;
    int qpos = 0;
    for (int i = 0; i < res.alignmentLength; i++) {
        if (qpos < flank_len) {
            if (res.alignment[i] == EDLIB_EDOP_MATCH) {
                qpos++;
            } else if (res.alignment[i] == EDLIB_EDOP_MISMATCH) {
                qpos++;
            } else if (res.alignment[i] == EDLIB_EDOP_DELETE) {
            } else if (res.alignment[i] == EDLIB_EDOP_INSERT) {
                qpos++;
            }
        } else {
            if (query_len == 0) {
                break;
            }
            if (res.alignment[i] == EDLIB_EDOP_MATCH) {
                query_len--;
            } else if (res.alignment[i] == EDLIB_EDOP_MISMATCH) {
                dist++;
                query_len--;
            } else if (res.alignment[i] == EDLIB_EDOP_DELETE) {
                dist += 1;
            } else if (res.alignment[i] == EDLIB_EDOP_INSERT) {
                dist += 1;
                query_len--;
            }
        }
    }
    return dist;
}

void log_mistake(const std::unordered_map<std::string, EdlibAlignResult>& scores,
                 const std::unordered_map<std::string, std::string>& strand_seq,
                 const std::unordered_map<std::string, std::string>& barcode_seq,
                 const std::string& pred_bc,
                 const std::string& real_bc) {
    spdlog::error(
            "\n{} dist {}, start {}, alan {}, seq {} bc {}\n{}\n{} dist {}, start {}, alen {}, seq "
            "{} bc {}\n{}",
            pred_bc, scores.at(pred_bc).editDistance, scores.at(pred_bc).startLocations[0],
            calculate_gaps(scores.at(pred_bc)), strand_seq.at(pred_bc), barcode_seq.at(pred_bc),
            utils::alignment_to_str(barcode_seq.at(pred_bc).c_str(), strand_seq.at(pred_bc).c_str(),
                                    scores.at(pred_bc)),
            real_bc, scores.at(real_bc).editDistance, scores.at(real_bc).startLocations[0],
            calculate_gaps(scores.at(real_bc)), strand_seq.at(real_bc), barcode_seq.at(real_bc),
            utils::alignment_to_str(barcode_seq.at(real_bc).c_str(), strand_seq.at(real_bc).c_str(),
                                    scores.at(real_bc)));
}

std::tuple<std::unordered_map<std::string, EdlibAlignResult>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
calculate_per_barcode_info(const std::string& read_seq,
                           const std::string& read_seq_rev,
                           const std::string& front_flank,
                           const std::string rear_flank,
                           const std::unordered_map<std::string, std::string>& barcodes,
                           const std::vector<std::string>& kit_bcs) {
    bool has_flanks = !front_flank.empty() || !rear_flank.empty();
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    // Want to extract edit distance for barcode region when flanks are present, so get alignment path.
    align_config.task = has_flanks ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;

    // Use 150 bp for front and rear of sequence.
    auto front = read_seq.substr(0, 150);
    auto rear = read_seq_rev.substr(0, 150);

    std::unordered_map<std::string, EdlibAlignResult> scores;
    std::unordered_map<std::string, std::string> strand_seq;
    std::unordered_map<std::string, std::string> barcode_seq;
    for (auto& bc : kit_bcs) {
        //for (auto& [bc, bc_seq] : barcodes) {
        // TODO: If there are flanking sequences, instead of using the edit distance
        // for the barcode + flanks, re-calculate the edit distance for just the
        // 24bp barcode region.
        auto bc_seq = barcodes.at(bc);
        auto updated_bc_seq = front_flank + bc_seq + rear_flank;
        EdlibAlignResult front_result = edlibAlign(updated_bc_seq.data(), updated_bc_seq.length(),
                                                   front.data(), front.length(), align_config);
        EdlibAlignResult rear_result = edlibAlign(updated_bc_seq.data(), updated_bc_seq.length(),
                                                  rear.data(), rear.length(), align_config);
        // Use the minimum edit distance alignment between front and rear matches.
        // Track the results for analysis and debugging.
        if (front_result.editDistance < rear_result.editDistance) {
            scores[bc] = front_result;
            strand_seq[bc] = front;
            barcode_seq[bc] = updated_bc_seq;
        } else {
            scores[bc] = rear_result;
            strand_seq[bc] = rear;
            barcode_seq[bc] = updated_bc_seq;
        }
    }

    return {scores, strand_seq, barcode_seq};
}

std::string Barcoder::edlib_barcode(const std::string& read_seq, const std::string& read_seq_rev) {
    std::string bc = UNCLASSIFIED_BARCODE;

    auto front = read_seq.substr(0, 150);
    auto rear = read_seq_rev.substr(0, 150);

    std::unordered_map<std::string, EdlibAlignResult> scores;
    std::unordered_map<std::string, std::string> strand_seq;
    std::unordered_map<std::string, std::string> barcode_seq;

    // Stage 1 - Find edit distance without alignment on raw barcode sequences withOUT flanks.
    std::tie(scores, strand_seq, barcode_seq) = calculate_per_barcode_info(
            front, rear, "", "", m_barcodes, barcoding::kit_info.at(m_kit_name).barcodes);

    // Find minimum edit distance found.
    auto min = std::min_element(scores.begin(), scores.end(), [](const auto& l, const auto& r) {
        return (l.second.editDistance < r.second.editDistance);
    });
    auto min_dist = min->second.editDistance;
    // Check how many barcodes achieved that edit distance.
    auto count_min = std::count_if(scores.begin(), scores.end(), [=](const auto& l) {
        return l.second.editDistance == min_dist;
    });

    // If edit distance meets threshold and there's only one barcode that meets the criteria, return it.
    int kMaxAllowedEditDistance = 5;

    // For debug only. Hard code the expected barcode so
    // the debug print can show edit distance and alignment
    // details to understand the false positives.
    const std::string kExpectedBarcode = "BC11";

    if (count_min == 1 && min_dist <= kMaxAllowedEditDistance) {
        bc = min->first;
        if (bc != kExpectedBarcode) {
            log_mistake(scores, strand_seq, barcode_seq, min->first, kExpectedBarcode);
        }
        return bc;
    }

    // Stage 2 - Find edit distance with alignment on barcode sequences WITH flanks.
    std::tie(scores, strand_seq, barcode_seq) = calculate_per_barcode_info(
            front, rear, barcoding::kit_info.at(m_kit_name).fwd_front_flank,
            barcoding::kit_info.at(m_kit_name).fwd_rear_flank, m_barcodes,
            barcoding::kit_info.at(m_kit_name).barcodes);

    // Find minimum edit distance found.
    min = std::min_element(scores.begin(), scores.end(), [](const auto& l, const auto& r) {
        return (l.second.editDistance < r.second.editDistance);
    });

    min_dist = min->second.editDistance;
    // Check how many barcodes achieved that edit distance.
    count_min = std::count_if(scores.begin(), scores.end(),
                              [=](const auto& l) { return l.second.editDistance == min_dist; });

    // If edit distance meets threshold and there's only one barcode that meets the criteria, return it.
    int kMaxAllowedEditDistanceWithFlanks = 18;
    if (count_min == 1 && min_dist <= kMaxAllowedEditDistanceWithFlanks) {
        bc = min->first;
        if (bc != kExpectedBarcode) {
            log_mistake(scores, strand_seq, barcode_seq, min->first, kExpectedBarcode);
        }
        return bc;
    } else {
        //    // Count is not 1, so check which one has least number of insertion/deletion errors.
        //    min = std::min_element(scores.begin(), scores.end(), [](const auto& l, const auto& r) {
        //        if (l.second.editDistance > r.second.editDistance)
        //            return false;
        //        if (l.second.editDistance == r.second.editDistance) {
        //            return calculate_gaps(l.second) < calculate_gaps(r.second);
        //        }
        //        return true;
        //    });
    }

    spdlog::debug("Min dist {}, count {}", min->second.editDistance, count_min);

    return bc;
}

std::string Barcoder::mm2_barcode(const std::string& seq,
                                  const std::string_view& qname,
                                  mm_tbuf_t* buf) {
    std::string barcode = UNCLASSIFIED_BARCODE;
    auto short_seq = seq.substr(0, 250);
    // do the mapping
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, short_seq.length(), short_seq.c_str(), &hits, buf, &m_map_opt,
                            qname.data());

    // just return the input record
    if (hits > 0) {
        auto best_map = std::max_element(
                reg, reg + hits,
                [&](const mm_reg1_t& a, const mm_reg1_t& b) { return a.mapq < b.mapq; });

        int32_t tid = best_map->rid;
        hts_pos_t qs = best_map->qs;
        hts_pos_t qe = best_map->qe;
        uint8_t mapq = best_map->mapq;

        if (hits > 1) {
            spdlog::debug("Found {} hits, best mapq {} qs {} qe {}, strand {}", hits, mapq, qs, qe,
                          best_map->rev ? '-' : '+');
        }
        if (!best_map->rev && mapq > m_q) {
            barcode = std::string(m_index->seq[best_map->rid].name);
        }

        free(best_map->p);
    }
    free(reg);
    return barcode;
}

std::vector<BamPtr> Barcoder::barcode(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    // get query name.
    std::string_view qname(bam_get_qname(irecord));

    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    // Pre-generate reverse complement sequence.
    std::string seq_rev = utils::reverse_complement(seq);

    //auto bc = mm2_barcode(seq, qname, buf);
    auto bc = edlib_barcode(seq, seq_rev);
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    if (bc != UNCLASSIFIED_BARCODE) {
        m_matched++;
    }
    results.push_back(BamPtr(bam_dup1(irecord)));

    return results;
}

stats::NamedStats Barcoder::sample_stats() const { return stats::from_obj(m_work_queue); }

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
void Barcoder::read_barcodes(const std::string& barcode_file) {
    std::cerr << __LINE__ << std::endl;
    std::cerr << barcode_file << std::endl;
    ifstream testFile(barcode_file);
    string bc_name;
    string bc_seq;
    string str;
    while (getline(testFile, str)) {
        std::cerr << str << std::endl;
        if (bc_name.empty()) {
            bc_name = str;
        } else {
            bc_seq = str;
            m_barcodes[bc_name] = bc_seq;
            bc_name = "";
            std::cerr << bc_name << ", " << bc_seq << "." << std::endl;
        }
    }

    spdlog::info("Number of barcodes {}", m_barcodes.size());
}

}  // namespace dorado
