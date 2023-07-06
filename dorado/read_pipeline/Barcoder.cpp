#include "Barcoder.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "htslib/sam.h"
#include "utils/alignment_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dorado {

const std::string UNCLASSIFIED_BARCODE = "unclassified";

BarcoderNode::BarcoderNode(MessageSink& sink,
                           const std::vector<std::string>& barcodes,
                           int threads,
                           const std::vector<std::string>& kit_names)
        : MessageSink(10000), m_sink(sink), m_threads(threads), m_barcoder(kit_names) {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&BarcoderNode::worker_thread, this, i)));
    }
}

BarcoderNode::~BarcoderNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    // Adding for thread safety in case worker thread throws exception.
    m_sink.terminate();
}

void BarcoderNode::worker_thread(size_t tid) {
    m_active++;  // Track active threads.

    Message message;
    while (m_work_queue.try_pop(message)) {
        auto read = std::get<BamPtr>(std::move(message));
        auto records = barcode(read.get());
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

std::vector<BamPtr> BarcoderNode::barcode(bam1_t* irecord) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;
    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);

    auto bc = m_barcoder.barcode(seq).adapter_name;
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

    for (auto& kit_name : final_kit_names) {
        auto kit_info = dorado::kit_info.at(kit_name);
        for (auto& bc_name : kit_info.barcodes) {
            AdapterSequence as;
            //as.adapter = m_barcodes.at(bc_name);
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

ScoreResults Barcoder::calculate_adapter_score(const std::string_view& read_seq,
                                               const std::string_view& read_seq_rev,
                                               const AdapterSequence& as,
                                               bool with_flanks) {
    std::vector<EdlibAlignResult> results;

    std::string_view temp_top = read_seq.substr(0, 150);
    std::string_view comp_bottom = read_seq_rev.substr(0, 150);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.mode = EDLIB_MODE_HW;
    align_config.task = EDLIB_TASK_LOC;

    if (!with_flanks) {
        // Calculate score for adapter match against template strand
        // Match top strand front
        EdlibAlignResult temp_top_strand =
                edlibAlign(as.adapter.data(), as.adapter.length(), temp_top.data(),
                           temp_top.length(), align_config);
        results.push_back(temp_top_strand);

        EdlibAlignResult comp_bottom_strand =
                edlibAlign(as.adapter.data(), as.adapter_rev.length(), comp_bottom.data(),
                           comp_bottom.length(), align_config);
        results.push_back(comp_bottom_strand);

        auto best_res = std::min_element(
                results.begin(), results.end(),
                [](const auto& l, const auto& r) { return l.editDistance < r.editDistance; });
        return {best_res->editDistance, as.adapter_name, as.kit};
    } else {
        // Calculate score for adapter match against template strand
        // Match top strand front
        EdlibAlignResult temp_top_strand =
                edlibAlign(as.top_primer.data(), as.top_primer.length(), temp_top.data(),
                           temp_top.length(), align_config);
        results.push_back(temp_top_strand);

        EdlibAlignResult comp_bottom_strand =
                edlibAlign(as.top_primer.data(), as.top_primer_rev.length(), comp_bottom.data(),
                           comp_bottom.length(), align_config);
        results.push_back(comp_bottom_strand);

        auto best_res = std::min_element(
                results.begin(), results.end(),
                [](const auto& l, const auto& r) { return l.editDistance < r.editDistance; });
        return {best_res->editDistance, as.adapter_name, as.kit};
    }
}

ScoreResults Barcoder::find_best_adapter(const std::string& read_seq,
                                         std::vector<AdapterSequence>& adapters) {
    std::vector<ScoreResults> scores;

    std::string fwd = read_seq;
    std::string rev = utils::reverse_complement(read_seq);

    for (auto& as : adapters) {
        scores.push_back(calculate_adapter_score(fwd, rev, as, false));
    }

    auto best_score =
            std::min_element(scores.begin(), scores.end(),
                             [](const auto& l, const auto& r) { return l.score < r.score; });
    auto count_min = std::count_if(scores.begin(), scores.end(), [best_score](const auto& l) {
        return l.score == best_score->score;
    });
    if (best_score->score <= 5 && count_min == 1) {
        return *best_score;
    }

    scores.clear();
    for (auto& as : adapters) {
        scores.push_back(calculate_adapter_score(fwd, rev, as, true));
    }

    best_score = std::min_element(scores.begin(), scores.end(),
                                  [](const auto& l, const auto& r) { return l.score < r.score; });
    count_min = std::count_if(scores.begin(), scores.end(),
                              [best_score](const auto& l) { return l.score == best_score->score; });
    if (best_score->score <= 18 && count_min == 1) {
        return *best_score;
    }
    return {100000, UNCLASSIFIED_BARCODE, UNCLASSIFIED_BARCODE};
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

}  // namespace dorado
