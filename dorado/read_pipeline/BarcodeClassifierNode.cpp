#include "BarcodeClassifierNode.h"

#include "BarcodeClassifier.h"
#include "utils/barcode_kits.h"
#include "htslib/sam.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

const std::string UNCLASSIFIED_BARCODE = "unclassified";

std::string generate_barcode_string(dorado::demux::ScoreResults bc_res) {
    auto bc = (bc_res.adapter_name == UNCLASSIFIED_BARCODE)
                      ? UNCLASSIFIED_BARCODE
                      : bc_res.kit + "_" + bc_res.adapter_name;
    spdlog::debug("BC: {}", bc);
    return bc;
}

}  // namespace

namespace dorado {

// A Node which encapsulates running barcode classification on each read.
BarcodeClassifierNode::BarcodeClassifierNode(int threads,
                                             const std::vector<std::string>& kit_names,
                                             bool barcode_both_ends,
                                             bool no_trim)
        : MessageSink(10000),
          m_threads(threads),
          m_barcoder(kit_names, barcode_both_ends),
          m_trim_barcodes(!no_trim) {
    start_threads();
}

void BarcodeClassifierNode::start_threads() {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(std::make_unique<std::thread>(
                std::thread(&BarcodeClassifierNode::worker_thread, this, i)));
    }
}

void BarcodeClassifierNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
}

void BarcodeClassifierNode::restart() {
    restart_input_queue();
    start_threads();
}

BarcodeClassifierNode::~BarcodeClassifierNode() { terminate_impl(); }

void BarcodeClassifierNode::worker_thread(size_t tid) {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            barcode(read);
            send_message_to_sink(std::move(read));
        } else if (std::holds_alternative<ReadPtr>(message)) {
            auto read = std::get<ReadPtr>(std::move(message));
            barcode(read);
            send_message_to_sink(std::move(read));
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

std::string trim_sequence(const std::string& seq, const std::pair<int, int> trim_interval) {
    int start_pos = trim_interval.first;
    int len = trim_interval.second - start_pos;
    return seq.substr(start_pos, len);
}

std::vector<uint8_t> extract_quality(bam1_t* input_record, int seqlen) {
    auto qual_aln = bam_get_qual(input_record);
    std::vector<uint8_t> qual;
    if (qual_aln) {
        qual = std::vector<uint8_t>(bam_get_qual(input_record),
                                    bam_get_qual(input_record) + seqlen);
    }
    return qual;
}

std::vector<uint8_t> trim_quality(const std::vector<uint8_t>& qual,
                                  const std::pair<int, int> trim_interval) {
    if (!qual.empty()) {
        return std::vector<uint8_t>(qual.begin() + trim_interval.first,
                                    qual.begin() + trim_interval.second);
    }
    return {};
}

std::vector<uint8_t> extract_move_table(bam1_t* input_record) {
    auto move_vals_aux = bam_aux_get(input_record, "mv");
    std::vector<uint8_t> move_vals;
    if (move_vals_aux) {
        int len = bam_auxB_len(move_vals_aux);
        move_vals.resize(len);
        for (int i = 0; i < len; i++) {
            move_vals[i] = bam_auxB2i(move_vals_aux, i);
        }
        bam_aux_del(input_record, move_vals_aux);
    }
    return move_vals;
}

std::tuple<size_t, std::vector<uint8_t>> trim_move_table(const std::vector<uint8_t>& move_vals,
                                                         const std::pair<int, int> trim_interval) {
    std::vector<uint8_t> trimmed_moves;
    size_t samples_trimmed = 0;
    if (!move_vals.empty()) {
        int stride = move_vals[0];
        trimmed_moves.push_back(stride);
        int bases_seen = -1;
        for (int i = 1; i < move_vals.size(); i++) {
            if (bases_seen >= trim_interval.second) {
                break;
            }
            auto mv = move_vals[i];
            if (mv == 1) {
                bases_seen++;
            }
            if (bases_seen >= trim_interval.first) {
                trimmed_moves.push_back(mv);
            } else {
                samples_trimmed += stride;
            }
        }
    }
    return {samples_trimmed, trimmed_moves};
}

std::tuple<std::string, std::vector<int8_t>> extract_modbase_info(bam1_t* input_record) {
    std::string modbase_str;
    std::vector<int8_t> modbase_probs;
    auto modbase_str_aux = bam_aux_get(input_record, "MM");
    if (modbase_str_aux) {
        modbase_str = std::string(bam_aux2Z(modbase_str_aux));
        bam_aux_del(input_record, modbase_str_aux);

        auto modbase_prob_aux = bam_aux_get(input_record, "ML");
        int len = bam_auxB_len(modbase_prob_aux);
        modbase_probs.resize(len);
        for (int i = 0; i < len; i++) {
            modbase_probs[i] = bam_auxB2i(modbase_prob_aux, i);
        }
        bam_aux_del(input_record, modbase_prob_aux);
    }

    return {modbase_str, modbase_probs};
}

std::tuple<std::string, std::vector<int8_t>> trim_modbase_info(
        const std::string& modbase_str,
        const std::vector<int8_t>& modbase_probs,
        const std::pair<int, int> trim_interval) {
    int start = trim_interval.first;
    int end = trim_interval.second;

    std::string trimmed_modbase_str;
    std::vector<int8_t> trimmed_modbase_probs;
    if (!modbase_str.empty()) {
        std::vector<std::pair<size_t, size_t>> delims;
        size_t pos = 0;
        while (pos < modbase_str.length()) {
            size_t delim_pos = modbase_str.find_first_of(';', pos);
            delims.push_back({pos, delim_pos});
            pos = delim_pos + 1;
        }
        size_t prob_pos = 0;
        for (auto [a, b] : delims) {
            std::string prefix = "";
            std::string counts = "";
            int bases_seen = 0;
            bool in_counts = false;
            pos = a;
            bool found_start = false;
            while (pos < b) {
                auto comma_pos = std::min(modbase_str.find_first_of(',', pos), b);
                auto substr_len = comma_pos - pos;
                if (!in_counts) {
                    in_counts = true;
                    prefix = modbase_str.substr(pos, substr_len);
                } else {
                    int num_skips = std::stoi(modbase_str.substr(pos, substr_len));
                    if (num_skips + bases_seen >= end) {
                        // Do nothing as these modbases are trimmed.
                    } else if (num_skips + bases_seen >= start) {
                        if (!found_start) {
                            counts += "," + std::to_string(num_skips + bases_seen - start);
                            found_start = true;
                        } else {
                            counts += "," + std::to_string(num_skips);
                        }
                        if (!modbase_probs.empty()) {
                            trimmed_modbase_probs.push_back(modbase_probs[prob_pos]);
                        }
                    }
                    prob_pos++;
                    bases_seen += num_skips + 1;
                }
                pos = comma_pos + 1;
            }
            if (!counts.empty()) {
                trimmed_modbase_str += prefix + counts + ";";
            }
        }
    }
    return {trimmed_modbase_str, trimmed_modbase_probs};
}

std::pair<int, int> determine_trim_interval(const demux::ScoreResults& res, int seqlen) {
    // Initialize interval to be the whole read. Note that the interval
    // defines which portion of the read to retain.
    std::pair<int, int> trim_interval = {0, seqlen};

    if (res.kit == UNCLASSIFIED_BARCODE) {
        return trim_interval;
    }

    const float kFlankScoreThres = 0.6f;

    // Use barcode flank positions to determine trim interval
    // only if the flanks were confidently found.
    auto kit_info_map = barcode_kits::get_kit_infos();
    const barcode_kits::KitInfo& kit = kit_info_map.at(res.kit);
    if (kit.double_ends) {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second;
        }

        float bottom_flank_score = res.bottom_flank_score;
        if (bottom_flank_score > kFlankScoreThres) {
            trim_interval.second = res.bottom_barcode_pos.first;
        }
    } else {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second;
        }
    }

    return trim_interval;
}

bam1_t* BarcodeClassifierNode::trim_barcode(bam1_t* input_record,
                                            const demux::ScoreResults& res,
                                            int seqlen) {
    auto trim_interval = determine_trim_interval(res, seqlen);

    if (trim_interval.second - trim_interval.first == seqlen) {
        return bam_dup1(input_record);
    }

    // Fetch components that need to be trimmed.
    auto bseq = bam_get_seq(input_record);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    std::vector<uint8_t> qual = extract_quality(input_record, seqlen);
    std::vector<uint8_t> move_vals = extract_move_table(input_record);
    int ts = bam_aux2i(bam_aux_get(input_record, "ts"));
    auto [modbase_str, modbase_probs] = extract_modbase_info(input_record);

    // Actually trim components.
    auto trimmed_seq = trim_sequence(seq, trim_interval);
    auto trimmed_qual = trim_quality(qual, trim_interval);
    auto [num_additional_samples_trimmed, trimmed_moves] =
            trim_move_table(move_vals, trim_interval);
    ts += num_additional_samples_trimmed;
    auto [trimmed_modbase_str, trimmed_modbase_probs] =
            trim_modbase_info(modbase_str, modbase_probs, trim_interval);

    // Create a new bam record to hold the trimmed read.
    bam1_t* out_record = bam_init1();
    bam_set1(out_record, input_record->core.l_qname - input_record->core.l_extranul - 1,
             bam_get_qname(input_record), input_record->core.flag, input_record->core.tid,
             input_record->core.pos, input_record->core.qual, input_record->core.n_cigar,
             bam_get_cigar(input_record), input_record->core.mtid, input_record->core.mpos,
             input_record->core.isize, trimmed_seq.size(), trimmed_seq.data(),
             trimmed_qual.empty() ? NULL : (char*)trimmed_qual.data(), bam_get_l_aux(input_record));
    memcpy(bam_get_aux(out_record), bam_get_aux(input_record), bam_get_l_aux(input_record));
    out_record->l_data += bam_get_l_aux(input_record);

    if (!trimmed_moves.empty()) {
        bam_aux_update_array(out_record, "mv", 'c', trimmed_moves.size(),
                             (uint8_t*)trimmed_moves.data());
    }

    if (!trimmed_modbase_str.empty()) {
        bam_aux_append(out_record, "MM", 'Z', trimmed_modbase_str.length() + 1,
                       (uint8_t*)trimmed_modbase_str.c_str());
        bam_aux_update_array(out_record, "ML", 'C', trimmed_modbase_probs.size(),
                             (uint8_t*)trimmed_modbase_probs.data());
    }

    bam_aux_update_int(out_record, "ts", ts);

    return out_record;
}

void BarcodeClassifierNode::trim_barcode(ReadPtr& read, const demux::ScoreResults& res) {
    int seqlen = read->seq.length();
    auto trim_interval = determine_trim_interval(res, seqlen);

    if (trim_interval.second - trim_interval.first == seqlen) {
        return;
    }

    read->seq = trim_sequence(read->seq, trim_interval);
    read->qstring = trim_sequence(read->qstring, trim_interval);
    size_t num_additional_samples_trimmed;
    std::tie(num_additional_samples_trimmed, read->moves) =
            trim_move_table(read->moves, trim_interval);
    read->num_trimmed_samples += num_additional_samples_trimmed;

    std::tie(read->modbase_bam_tag, read->modbase_probs_bam_tag) =
            trim_modbase_info(read->modbase_bam_tag, read->modbase_probs_bam_tag, trim_interval);
}

void BarcodeClassifierNode::barcode(BamPtr& read) {
    bam1_t* irecord = read.get();
    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;
    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);

    auto bc_res = m_barcoder.barcode(seq);
    auto bc = generate_barcode_string(bc_res);
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    m_num_records++;

    if (m_trim_barcodes) {
        read = BamPtr(trim_barcode(irecord, bc_res, seqlen));
    }
}

void BarcodeClassifierNode::barcode(ReadPtr& read) {
    // get the sequence to map from the record
    auto bc_res = m_barcoder.barcode(read->seq);
    auto bc = generate_barcode_string(bc_res);
    read->barcode = bc;
    m_num_records++;

    if (m_trim_barcodes) {
        trim_barcode(read, bc_res);
    }
}

stats::NamedStats BarcodeClassifierNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_barcodes_demuxed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
