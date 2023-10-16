#include "BarcodeClassifierNode.h"

#include "demux/BarcodeClassifier.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
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
    std::string bc;
    if (bc_res.adapter_name != UNCLASSIFIED_BARCODE) {
        bc = dorado::barcode_kits::generate_standard_barcode_name(bc_res.kit, bc_res.adapter_name);
    } else {
        bc = UNCLASSIFIED_BARCODE;
    }
    spdlog::debug("BC: {}", bc);
    return bc;
}

}  // namespace

namespace dorado {

// A Node which encapsulates running barcode classification on each read.
BarcodeClassifierNode::BarcodeClassifierNode(int threads,
                                             const std::vector<std::string>& kit_names,
                                             bool barcode_both_ends,
                                             bool no_trim,
                                             const BarcodingInfo::FilterSet& allowed_barcodes)
        : MessageSink(10000),
          m_threads(threads),
          m_default_barcoding_info(
                  create_barcoding_info(kit_names, barcode_both_ends, !no_trim, allowed_barcodes)) {
    start_threads();
}

BarcodeClassifierNode::BarcodeClassifierNode(int threads) : MessageSink(10000), m_threads(threads) {
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
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            barcode(*read);
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

        // In some cases where the read length is very small, the front
        // and rear windows could actually overlap. In that case find
        // which window was used and just grab the interval for that
        // window.
        if (trim_interval.second <= trim_interval.first) {
            if (res.use_top) {
                return {res.top_barcode_pos.first, res.top_barcode_pos.second + 1};
            } else {
                return {res.bottom_barcode_pos.first, res.bottom_barcode_pos.second + 1};
            }
        }
    } else {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second + 1;
        }
    }

    if (trim_interval.second <= trim_interval.first) {
        // This could happen if the read is very short and the barcoding
        // algorithm determines the barcode interval to be the entire read.
        // In that case, skip trimming.
        trim_interval = {0, seqlen};
    }

    return trim_interval;
}

BamPtr BarcodeClassifierNode::trim_barcode(BamPtr input,
                                           const demux::ScoreResults& res,
                                           int seqlen) {
    auto trim_interval = determine_trim_interval(res, seqlen);

    if (trim_interval.second - trim_interval.first == seqlen) {
        return input;
    }

    bam1_t* input_record = input.get();

    // Fetch components that need to be trimmed.
    std::string seq = utils::extract_sequence(input_record, seqlen);
    std::vector<uint8_t> qual = utils::extract_quality(input_record, seqlen);
    auto [stride, move_vals] = utils::extract_move_table(input_record);
    int ts = bam_aux_get(input_record, "ts") ? bam_aux2i(bam_aux_get(input_record, "ts")) : 0;
    auto [modbase_str, modbase_probs] = utils::extract_modbase_info(input_record);

    // Actually trim components.
    auto trimmed_seq = utils::trim_sequence(seq, trim_interval);
    auto trimmed_qual = utils::trim_quality(qual, trim_interval);
    auto [positions_trimmed, trimmed_moves] = utils::trim_move_table(move_vals, trim_interval);
    ts += positions_trimmed * stride;
    auto [trimmed_modbase_str, trimmed_modbase_probs] =
            utils::trim_modbase_info(seq, modbase_str, modbase_probs, trim_interval);

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

    return BamPtr(out_record);
}

void BarcodeClassifierNode::trim_barcode(SimplexRead& read, const demux::ScoreResults& res) {
    int seqlen = read.read_common.seq.length();
    auto trim_interval = determine_trim_interval(res, seqlen);

    if (trim_interval.second - trim_interval.first == seqlen) {
        return;
    }

    read.read_common.seq = utils::trim_sequence(read.read_common.seq, trim_interval);
    read.read_common.qstring = utils::trim_sequence(read.read_common.qstring, trim_interval);
    size_t num_positions_trimmed;
    std::tie(num_positions_trimmed, read.read_common.moves) =
            utils::trim_move_table(read.read_common.moves, trim_interval);
    read.read_common.num_trimmed_samples += read.read_common.model_stride * num_positions_trimmed;

    if (read.read_common.mod_base_info) {
        int num_modbase_channels = read.read_common.mod_base_info->alphabet.size();
        // The modbase probs table consists of the probability per channel per base. So when
        // trimming, we just shift everything by skipped bases * number of channels.
        std::pair<int, int> modbase_interval = {trim_interval.first * num_modbase_channels,
                                                trim_interval.second * num_modbase_channels};
        read.read_common.base_mod_probs =
                utils::trim_quality(read.read_common.base_mod_probs, modbase_interval);
    }
}

std::shared_ptr<const BarcodingInfo> BarcodeClassifierNode::get_barcoding_info(
        const SimplexRead& read) const {
    if (m_default_barcoding_info && !m_default_barcoding_info->kit_name.empty()) {
        return m_default_barcoding_info;
    }

    if (read.read_common.barcoding_info && !read.read_common.barcoding_info->kit_name.empty()) {
        return read.read_common.barcoding_info;
    }

    return nullptr;
}

void BarcodeClassifierNode::barcode(BamPtr& read) {
    if (!m_default_barcoding_info || m_default_barcoding_info->kit_name.empty()) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(m_default_barcoding_info->kit_name);

    bam1_t* irecord = read.get();
    int seqlen = irecord->core.l_qseq;
    std::string seq = utils::extract_sequence(irecord, seqlen);

    auto bc_res = barcoder->barcode(seq, m_default_barcoding_info->barcode_both_ends);
    auto bc = generate_barcode_string(bc_res);
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    m_num_records++;

    if (m_default_barcoding_info->trim) {
        read = trim_barcode(std::move(read), bc_res, seqlen);
    }
}

void BarcodeClassifierNode::barcode(SimplexRead& read) {
    auto barcoding_info = get_barcoding_info(read);
    if (!barcoding_info) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(barcoding_info->kit_name);

    // get the sequence to map from the record
    auto bc_res = barcoder->barcode(read.read_common.seq, barcoding_info->barcode_both_ends);
    read.read_common.barcode = generate_barcode_string(bc_res);
    m_num_records++;
    if (barcoding_info->trim) {
        trim_barcode(read, bc_res);
    }
}

stats::NamedStats BarcodeClassifierNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_barcodes_demuxed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
