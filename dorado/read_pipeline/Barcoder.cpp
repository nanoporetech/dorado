#include "Barcoder.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "htslib/sam.h"
#include "utils/alignment_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace {

// Calculate the edit distance for an alignment just within the region
// which maps to the barcode sequence. i.e. Ignore any edits made to the
// flanking regions.
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
            //std::cerr << qpos << ", " << i << std::endl;
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
                query_len--;
                dist += 1;
            }
        }
    }
    return dist;
}

}  // namespace

const std::string UNCLASSIFIED_BARCODE = "unclassified";

BarcoderNode::BarcoderNode(int threads, const std::vector<std::string>& kit_names)
        : MessageSink(10000), m_threads(threads), m_barcoder(kit_names) {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&BarcoderNode::worker_thread, this, i)));
    }
}

void BarcoderNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
}

BarcoderNode::~BarcoderNode() {
    terminate_impl();
    spdlog::debug("> Barcoded: {}", m_matched.load());
}

void BarcoderNode::worker_thread(size_t tid) {
    Message message;
    while (m_work_queue.try_pop(message)) {
        auto read = std::get<BamPtr>(std::move(message));
        auto records = barcode(read.get());
        for (auto& record : records) {
            send_message_to_sink(std::move(record));
        }
    }
}

std::vector<BamPtr> BarcoderNode::barcode(bam1_t* irecord) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;
    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);

    auto bc_res = m_barcoder.barcode(seq);
    auto bc = (bc_res.adapter_name == UNCLASSIFIED_BARCODE)
                      ? UNCLASSIFIED_BARCODE
                      : bc_res.kit + "_" + bc_res.adapter_name;
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    if (bc != UNCLASSIFIED_BARCODE) {
        m_matched++;
    }
    results.push_back(BamPtr(bam_dup1(irecord)));

    return results;
}

stats::NamedStats BarcoderNode::sample_stats() const { return stats::from_obj(m_work_queue); }

Barcoder::Barcoder(const std::vector<std::string>& kit_names) {
    m_adapter_sequences = generate_adapter_sequence(kit_names);
}

ScoreResults Barcoder::barcode(const std::string& seq) {
    auto best_adapter = find_best_adapter(seq, m_adapter_sequences);
    return best_adapter;
}

// Generate all possible barcode adapters. If kit name is passed
// limit the adapters generated to only the specified kits.
// Returns a vector all barcode adapter sequences to test the
// input read sequence against.
std::vector<AdapterSequence> Barcoder::generate_adapter_sequence(
        const std::vector<std::string>& kit_names) {
    std::vector<AdapterSequence> adapters;
    std::vector<std::string> final_kit_names;
    if (kit_names.size() == 0) {
        for (auto& [kit_name, kit] : kit_info) {
            final_kit_names.push_back(kit_name);
        }
    } else {
        final_kit_names = kit_names;
    }
    spdlog::debug("> Kits to evaluate: {}", final_kit_names.size());

    for (auto& kit_name : final_kit_names) {
        auto kit_info = dorado::kit_info.at(kit_name);
        for (auto& bc_name : kit_info.barcodes) {
            AdapterSequence as;
            as.adapter = barcodes.at(bc_name);
            as.adapter_rev = utils::reverse_complement(as.adapter);

            as.top_primer = kit_info.top_front_flank + as.adapter + kit_info.top_rear_flank;
            as.top_primer_rev = utils::reverse_complement(as.top_primer);
            as.bottom_primer =
                    kit_info.bottom_front_flank + as.adapter + kit_info.bottom_rear_flank;
            as.bottom_primer_rev = utils::reverse_complement(as.bottom_primer);

            as.top_primer_flank_len = kit_info.top_front_flank.length();
            as.bottom_primer_flank_len =
                    (kit_info.ends_different ? kit_info.bottom_front_flank.length()
                                             : as.top_primer_flank_len);
            ;

            as.adapter_name = bc_name;
            as.kit = kit_name;
            adapters.push_back(as);
        }
    }
    return adapters;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====----------------=====-> 3'
//      BCXX_1             RC(BCXX_2)
//
// 3' <-=====----------------=====-< 5'
//    RC(BCXX_1)             BCXX_2
//
// In this scenario, the barcode (and its flanks) ligate to both ends
// of the read. The adapter sequence is also different for top and bottom strands.
// So we need to check bottom ends of the read. Since the adapters always ligate to
// 5' end of the read, the 3' end of the other strand has the reverse complement
// of that adapter sequence.
ScoreResults Barcoder::calculate_adapter_score_different_double_ends(
        const std::string_view& read_seq,
        const AdapterSequence& as,
        bool with_flanks) {
    // This calculates the score for barcodes which ligate to both ends
    // of the strand.
    std::string_view read_top = read_seq.substr(0, 150);
    std::string_view read_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = EDLIB_TASK_LOC;

    // Track the score for each variant.
    // v1 = BCXX_1 ---- RC(BCXX_2)
    // v2 = BCXX_2 ---- RC(BCXX_1)
    std::string_view top_strand_v1;
    std::string_view bottom_strand_v1;
    std::string_view top_strand_v2;
    std::string_view bottom_strand_v2;
    if (with_flanks) {
        top_strand_v1 = as.top_primer;
        bottom_strand_v1 = as.bottom_primer_rev;
        top_strand_v2 = as.bottom_primer;
        bottom_strand_v2 = as.top_primer_rev;
    } else {
        top_strand_v1 = as.adapter;
        bottom_strand_v1 = as.adapter_rev;
        top_strand_v2 = as.adapter;
        bottom_strand_v2 = as.adapter_rev;
    }

    // Score for each variant is the average edit distance score for the
    // top and bottom windows.
    ScoreResults v1;
    EdlibAlignResult read_top_aln = edlibAlign(top_strand_v1.data(), top_strand_v1.length(),
                                               read_top.data(), read_top.length(), align_config);
    float top_match_factor = 1.f - (float)read_top_aln.editDistance / top_strand_v1.length();
    v1.top_score = top_match_factor;

    EdlibAlignResult read_bottom_aln =
            edlibAlign(bottom_strand_v1.data(), bottom_strand_v1.length(), read_bottom.data(),
                       read_bottom.length(), align_config);
    float bottom_match_factor =
            1.f - (float)read_bottom_aln.editDistance / bottom_strand_v1.length();
    v1.bottom_score = bottom_match_factor;
    v1.score = 0.5f * (v1.top_score + v1.bottom_score);

    ScoreResults v2;
    EdlibAlignResult read_top_aln_v2 = edlibAlign(top_strand_v2.data(), top_strand_v2.length(),
                                                  read_top.data(), read_top.length(), align_config);
    float top_match_factor_v2 = 1.f - (float)read_top_aln_v2.editDistance / top_strand_v2.length();
    v2.top_score = top_match_factor_v2;

    EdlibAlignResult read_bottom_aln_v2 =
            edlibAlign(bottom_strand_v2.data(), bottom_strand_v2.length(), read_bottom.data(),
                       read_bottom.length(), align_config);
    float bottom_match_factor_v2 =
            1.f - (float)read_bottom_aln_v2.editDistance / bottom_strand_v2.length();
    v2.bottom_score = bottom_match_factor_v2;
    v2.score = 0.5f * (v2.top_score + v2.bottom_score);

    // Final score is the minimum of the 2 variants.
    ScoreResults res;
    if (v1.score > v2.score) {
        res = v1;
    } else {
        res = v2;
    }
    res.adapter_name = as.adapter_name;
    res.kit = as.kit;
    return res;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====--------------=====-> 3'
//      BCXXX            RC(BCXXX)
//
// 3' <-=====--------------=====-< 5'
//    RC(BCXXX)           (BCXXX)
//
// In this scenario, the barcode (and its flanks) potentially ligate to both ends
// of the read. But the adapter sequence is the same for both top and bottom strands.
// So we need to check bottom ends of the read. However since adapter sequence is the
// same for top and bottom strands, we simply need to look for the adapter and its
// reverse complement sequence in the top/bottom windows.
ScoreResults Barcoder::calculate_adapter_score_double_ends(const std::string_view& read_seq,
                                                           const AdapterSequence& as,
                                                           bool with_flanks) {
    std::string_view read_top = read_seq.substr(0, 150);
    std::string_view read_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = (with_flanks ? EDLIB_TASK_PATH : EDLIB_TASK_PATH);

    std::string_view top_strand;
    std::string_view bottom_strand;
    if (with_flanks) {
        top_strand = as.top_primer;
        bottom_strand = as.top_primer_rev;
    } else {
        top_strand = as.adapter;
        bottom_strand = as.adapter_rev;
    }

    // Common score struct to track the score for both the top and bottom windows.
    ScoreResults res;
    res.adapter_name = as.adapter_name;
    res.kit = as.kit;

    auto scorer = [&as, &align_config, &with_flanks](const std::string_view& primer,
                                                     const std::string_view& read,
                                                     int flank_len) -> std::pair<float, float> {
        EdlibAlignResult aln = edlibAlign(primer.data(), primer.length(), read.data(),
                                          read.length(), align_config);
        float flank_score = -1.f;
        int adapter_edit_dist = aln.editDistance;
        spdlog::debug("{}: {}, {}", as.adapter_name, primer, read);
        if (with_flanks) {
            // Calculate edit distance of just the adapter portion without flanks.
            int primer_edit_dist = adapter_edit_dist;
            adapter_edit_dist = calculate_edit_dist(aln, flank_len, as.adapter.length());
            spdlog::debug("Top Full flank dist {}, no flank dist {}", primer_edit_dist,
                          adapter_edit_dist);
            flank_score = 1.f - ((float)primer_edit_dist - adapter_edit_dist) /
                                        (primer.length() - as.adapter.length());
        } else {
            spdlog::debug("Top No flank dist {}", adapter_edit_dist);
        }
        spdlog::debug("\n{}", utils::alignment_to_str(primer.data(), read.data(), aln));
        float adapter_score = 1.f - (float)adapter_edit_dist / as.adapter.length();
        return {adapter_score, flank_score};
    };

    // Generate top window score.
    std::tie(res.top_score, res.top_flank_score) =
            scorer(top_strand, read_top, as.top_primer_flank_len);
    // Generate top window score.
    std::tie(res.bottom_score, res.bottom_flank_score) =
            scorer(bottom_strand, read_bottom, as.bottom_primer_flank_len);

    // Then choose the window with the best score.
    if (res.top_score > res.bottom_score) {
        res.score = res.top_score;
        res.flank_score = res.top_flank_score;
    } else {
        res.score = res.bottom_score;
        res.flank_score = res.bottom_flank_score;
    }
    return res;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====---------------> 3'
//      BCXXX
//
// In this scenario, the barcode (and its flanks) only ligate to the 5' end
// of the read. So we only look for adapter sequence in the top "window" (first
// 150bp) of the read.
ScoreResults Barcoder::calculate_adapter_score(const std::string_view& read_seq,
                                               const AdapterSequence& as,
                                               bool with_flanks) {
    std::string_view temp_top = read_seq.substr(0, 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = EDLIB_TASK_LOC;

    std::string_view top_strand;
    if (with_flanks) {
        top_strand = as.top_primer;
    } else {
        top_strand = as.adapter;
    }

    // Calculate score for the raw barcode sequence (without flanks)
    // against the front and back windows. Return the best match found.
    EdlibAlignResult temp_top_strand = edlibAlign(top_strand.data(), top_strand.length(),
                                                  temp_top.data(), temp_top.length(), align_config);
    float score = 1.f - (float)temp_top_strand.editDistance / top_strand.length();
    ScoreResults res;
    res.score = score;
    res.top_score = score;
    res.bottom_score = score;
    return res;
}

// Score every barcode against the input read and returns the best match,
// or an unclassified match, based on certain heuristics.
ScoreResults Barcoder::find_best_adapter(const std::string& read_seq,
                                         std::vector<AdapterSequence>& adapters) {
    std::string fwd = read_seq;

    // Attempt to match the barcodes without and then with flanks.
    std::array<bool, 2> use_flanks{false, true};
    for (auto& use_flank : use_flanks) {
        std::vector<ScoreResults> scores;
        for (auto& as : adapters) {
            auto& kit = kit_info.at(as.kit);
            if (kit.double_ends) {
                if (kit.ends_different) {
                    scores.push_back(
                            calculate_adapter_score_different_double_ends(fwd, as, use_flank));
                } else {
                    scores.push_back(calculate_adapter_score_double_ends(fwd, as, use_flank));
                }
            } else {
                scores.push_back(calculate_adapter_score(fwd, as, use_flank));
            }
        }

        // Sore the scores windows by their adapter score.
        std::sort(scores.begin(), scores.end(),
                  [](const auto& l, const auto& r) { return l.score > r.score; });
        auto best_score = scores.begin();
        // At minimum, the best window must meet the adapter score threshold.
        const float kThres = 0.78f;
        if (best_score->score >= kThres) {
            // If there's only one window and it meets the threshold, choose it.
            if (scores.size() == 1) {
                return *best_score;
            } else {
                // Choose the best if it's sufficiently better than the second best score.
                auto second_best_score = std::next(scores.begin());
                auto& best_kit = kit_info.at(best_score->kit);
                auto& second_best_kit = kit_info.at(second_best_score->kit);
                if (best_kit.double_ends && second_best_kit.double_ends && use_flank) {
                    // If the best and 2nd best scores both are double ended adapters and
                    // we have the flank scores, choose the best only it has better adapter
                    // AND flank scores.
                    if (best_score->flank_score >= second_best_score->flank_score) {
                        spdlog::debug(
                                "Use flank {}: Best score {} (flank {}) 2nd best score {} (flank "
                                "{})",
                                use_flank, best_score->score, best_score->flank_score,
                                second_best_score->score, second_best_score->flank_score);
                        return *best_score;
                    }
                } else {
                    // Pick the best score only if it's better than the 2nd best score by a margin.
                    const float kMargin = 0.1f;
                    if (std::abs(best_score->score - second_best_score->score) >= kMargin) {
                        spdlog::debug("Use flank {}: Best score {} 2nd best score {}", use_flank,
                                      best_score->score, second_best_score->score);
                        return *best_score;
                    }
                }
            }
        }
    }

    // If nothing is found, report as unclassified.
    return {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, UNCLASSIFIED_BARCODE, UNCLASSIFIED_BARCODE};
}

}  // namespace dorado
