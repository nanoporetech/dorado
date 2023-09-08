#include "BarcodeClassifierNode.h"

#include "BarcodeClassifier.h"
#include "utils/barcode_kits.h"
#include "utils/bam_utils.h"
#include "utils/trim.h"
#include "utils/types.h"

#include <htslib/sam.h>
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

std::pair<int, int> determine_trim_interval(const demux::ScoreResults& res, int seqlen) {
    // Initialize interval to be the whole read. Note that the interval
    // defines which portion of the read to retain.
    std::pair<int, int> trim_interval = {0, seqlen};

    if (res.kit == UNCLASSIFIED_BARCODE) {
        return trim_interval;
    }

    const float kFlankScoreThres = 0.6f;

    // Use barcode flank positions to determine trim interval
    // only if the flanks were confidently found. 1 is added to
    // the end of top barcode end value because that's the position
    // in the sequence where the barcode ends. So the actual sequence
    // because from one after that.
    auto kit_info_map = barcode_kits::get_kit_infos();
    const barcode_kits::KitInfo& kit = kit_info_map.at(res.kit);
    if (kit.double_ends) {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second + 1;
        }

        float bottom_flank_score = res.bottom_flank_score;
        if (bottom_flank_score > kFlankScoreThres) {
            trim_interval.second = res.bottom_barcode_pos.first;
        }
    } else {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second + 1;
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
    std::string seq = utils::extract_sequence(input_record, seqlen);
    std::vector<uint8_t> qual = utils::extract_quality(input_record, seqlen);
    auto [stride, move_vals] = utils::extract_move_table(input_record);
    int ts = bam_aux2i(bam_aux_get(input_record, "ts"));
    auto [modbase_str, modbase_probs] = utils::extract_modbase_info(input_record);

    // Actually trim components.
    auto trimmed_seq = utils::trim_sequence(seq, trim_interval);
    auto trimmed_qual = utils::trim_quality(qual, trim_interval);
    auto [positions_trimmed, trimmed_moves] = utils::trim_move_table(move_vals, trim_interval);
    ts += positions_trimmed * stride;
    auto [trimmed_modbase_str, trimmed_modbase_probs] =
            utils::trim_modbase_info(modbase_str, modbase_probs, trim_interval);

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

    // Insert the new tags and delete the old ones.
    if (!trimmed_moves.empty()) {
        bam_aux_del(out_record, bam_aux_get(out_record, "mv"));
        // Move table format is stride followed by moves.
        trimmed_moves.insert(trimmed_moves.begin(), stride);
        bam_aux_update_array(out_record, "mv", 'c', trimmed_moves.size(),
                             (uint8_t*)trimmed_moves.data());
    }

    if (!trimmed_modbase_str.empty()) {
        bam_aux_del(out_record, bam_aux_get(out_record, "MM"));
        bam_aux_append(out_record, "MM", 'Z', trimmed_modbase_str.length() + 1,
                       (uint8_t*)trimmed_modbase_str.c_str());
        bam_aux_del(out_record, bam_aux_get(out_record, "ML"));
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

    read->seq = utils::trim_sequence(read->seq, trim_interval);
    read->qstring = utils::trim_sequence(read->qstring, trim_interval);
    size_t num_positions_trimmed;
    std::tie(num_positions_trimmed, read->moves) =
            utils::trim_move_table(read->moves, trim_interval);
    read->num_trimmed_samples += read->model_stride * num_positions_trimmed;

    std::tie(read->modbase_bam_tag, read->modbase_probs_bam_tag) = utils::trim_modbase_info(
            read->modbase_bam_tag, read->modbase_probs_bam_tag, trim_interval);
}

void BarcodeClassifierNode::barcode(BamPtr& read) {
    bam1_t* irecord = read.get();
    int seqlen = irecord->core.l_qseq;
    std::string seq = utils::extract_sequence(irecord, seqlen);

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
