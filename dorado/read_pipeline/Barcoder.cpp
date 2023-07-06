#include "Barcoder.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "htslib/sam.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

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
            as.bottom_primer_flank_len = kit_info.bottom_front_flank.length();

            as.adapter_name = bc_name;
            as.kit = kit_name;
            adapters.push_back(as);
        }
    }
    return adapters;
}

// Calculate a score for each barcode against the front and/or back "windows"
// of a read. A window is a segment of 150bp at the beginning or end of a read.
// The score returned is the edit distance to convert the adapter into the window.
ScoreResults Barcoder::calculate_adapter_score_different_double_ends(
        const std::string_view& read_seq,
        const AdapterSequence& as,
        bool with_flanks) {
    // This calculates the score for barcodes which ligate to both ends
    // of the strand.
    std::string_view temp_top = read_seq.substr(0, 150);
    std::string_view temp_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = EDLIB_TASK_LOC;

    std::string_view top_strand;
    std::string_view bottom_strand;
    if (with_flanks) {
        top_strand = as.top_primer;
        bottom_strand = as.bottom_primer_rev;
    } else {
        top_strand = as.adapter;
        bottom_strand = as.adapter_rev;
    }

    // Calculate score for the raw barcode sequence (without flanks)
    // against the front and back windows. Return the best match found.
    EdlibAlignResult temp_top_strand = edlibAlign(top_strand.data(), top_strand.length(),
                                                  temp_top.data(), temp_top.length(), align_config);
    float top_match_factor = 1.f - (float)temp_top_strand.editDistance / top_strand.length();

    EdlibAlignResult comp_bottom_strand =
            edlibAlign(bottom_strand.data(), bottom_strand.length(), temp_bottom.data(),
                       temp_bottom.length(), align_config);
    float bottom_match_factor =
            1.f - (float)comp_bottom_strand.editDistance / bottom_strand.length();

    return {0.5f * (top_match_factor + bottom_match_factor), as.adapter_name, as.kit};
}

// Calculate a score for each barcode against the front and/or back "windows"
// of a read. A window is a segment of 150bp at the beginning or end of a read.
// The score returned is the edit distance to convert the adapter into the window.
ScoreResults Barcoder::calculate_adapter_score_double_ends(const std::string_view& read_seq,
                                                           const AdapterSequence& as,
                                                           bool with_flanks) {
    // This calculates the score for barcodes which ligate to both ends
    // of the strand.
    std::vector<EdlibAlignResult> results;

    std::string_view temp_top = read_seq.substr(0, 150);
    std::string_view temp_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = EDLIB_TASK_LOC;

    std::string_view top_strand;
    std::string_view bottom_strand;
    if (with_flanks) {
        top_strand = as.top_primer;
        bottom_strand = as.top_primer_rev;
    } else {
        top_strand = as.adapter;
        bottom_strand = as.adapter_rev;
    }

    // Calculate score for the raw barcode sequence (without flanks)
    // against the front and back windows. Return the best match found.
    EdlibAlignResult temp_top_strand = edlibAlign(top_strand.data(), top_strand.length(),
                                                  temp_top.data(), temp_top.length(), align_config);
    results.push_back(temp_top_strand);

    EdlibAlignResult comp_bottom_strand =
            edlibAlign(bottom_strand.data(), bottom_strand.length(), temp_bottom.data(),
                       temp_bottom.length(), align_config);
    results.push_back(comp_bottom_strand);

    auto best_res = std::min_element(
            results.begin(), results.end(),
            [](const auto& l, const auto& r) { return l.editDistance < r.editDistance; });
    return {1.f - (float)best_res->editDistance / top_strand.length(), as.adapter_name, as.kit};
}

// Calculate a score for each barcode against the front and/or back "windows"
// of a read. A window is a segment of 150bp at the beginning or end of a read.
// The score returned is the edit distance to convert the adapter into the window.
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
    return {1.f - (float)temp_top_strand.editDistance / top_strand.length(), as.adapter_name,
            as.kit};
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

        auto best_score =
                std::max_element(scores.begin(), scores.end(),
                                 [](const auto& l, const auto& r) { return l.score < r.score; });
        auto count_min = std::count_if(scores.begin(), scores.end(), [best_score](const auto& l) {
            return l.score == best_score->score;
        });
        if (best_score->score >= 0.75f && count_min == 1) {
            return *best_score;
        }
    }

    // If nothing is found, report as unclassified.
    return {-1.f, UNCLASSIFIED_BARCODE, UNCLASSIFIED_BARCODE};
}

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

}  // namespace dorado
