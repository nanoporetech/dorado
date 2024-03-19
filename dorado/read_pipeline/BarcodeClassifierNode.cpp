#include "BarcodeClassifierNode.h"

#include "demux/BarcodeClassifier.h"
#include "demux/Trimmer.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/trim.h"
#include "utils/types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {

const std::string UNCLASSIFIED_BARCODE = "unclassified";

std::string generate_barcode_string(dorado::BarcodeScoreResult bc_res) {
    std::string bc;
    if (bc_res.barcode_name != UNCLASSIFIED_BARCODE) {
        bc = dorado::barcode_kits::generate_standard_barcode_name(bc_res.kit, bc_res.barcode_name);
    } else {
        bc = UNCLASSIFIED_BARCODE;
    }
    spdlog::trace("BC: {}", bc);
    return bc;
}

}  // namespace

namespace dorado {

// A Node which encapsulates running barcode classification on each read.
BarcodeClassifierNode::BarcodeClassifierNode(int threads,
                                             const std::vector<std::string>& kit_names,
                                             bool barcode_both_ends,
                                             bool no_trim,
                                             BarcodingInfo::FilterSet allowed_barcodes,
                                             const std::optional<std::string>& custom_kit,
                                             const std::optional<std::string>& custom_seqs)
        : MessageSink(10000, threads),
          m_default_barcoding_info(create_barcoding_info(kit_names,
                                                         barcode_both_ends,
                                                         !no_trim,
                                                         std::move(allowed_barcodes),
                                                         custom_kit,
                                                         custom_seqs)) {
    if (m_default_barcoding_info->kit_name.empty()) {
        spdlog::debug("Barcode with new kit from {}", *m_default_barcoding_info->custom_kit);
    } else {
        spdlog::debug("Barcode for {}", m_default_barcoding_info->kit_name);
    }
    start_input_processing(&BarcodeClassifierNode::input_thread_fn, this);
}

BarcodeClassifierNode::BarcodeClassifierNode(int threads) : MessageSink(10000, threads) {
    start_input_processing(&BarcodeClassifierNode::input_thread_fn, this);
}

void BarcodeClassifierNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            // If the read is a secondary or supplementary read, ignore it.
            if (read->core.flag & (BAM_FSUPPLEMENTARY | BAM_FSECONDARY)) {
                continue;
            }
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

std::shared_ptr<const BarcodingInfo> BarcodeClassifierNode::get_barcoding_info(
        const SimplexRead& read) const {
    if (m_default_barcoding_info && (!m_default_barcoding_info->kit_name.empty() ||
                                     m_default_barcoding_info->custom_kit.has_value())) {
        return m_default_barcoding_info;
    }

    if (read.read_common.barcoding_info &&
        (!read.read_common.barcoding_info->kit_name.empty() ||
         read.read_common.barcoding_info->custom_kit.has_value())) {
        return read.read_common.barcoding_info;
    }

    return nullptr;
}

void BarcodeClassifierNode::barcode(BamPtr& read) {
    if (!m_default_barcoding_info ||
        (m_default_barcoding_info->kit_name.empty() && !m_default_barcoding_info->custom_kit)) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(*m_default_barcoding_info);

    bam1_t* irecord = read.get();
    std::string seq = utils::extract_sequence(irecord);

    auto bc_res = barcoder->barcode(seq, m_default_barcoding_info->barcode_both_ends,
                                    m_default_barcoding_info->allowed_barcodes);
    auto bc = generate_barcode_string(bc_res);
    bam_aux_append(irecord, "BC", 'Z', int(bc.length() + 1), (uint8_t*)bc.c_str());
    m_num_records++;
    {
        std::lock_guard lock(m_barcode_count_mutex);
        m_barcode_count[bc]++;
    }

    if (m_default_barcoding_info->trim) {
        int seqlen = irecord->core.l_qseq;
        auto trim_interval = Trimmer::determine_trim_interval(bc_res, seqlen);

        if (trim_interval.second - trim_interval.first < seqlen) {
            read = Trimmer::trim_sequence(std::move(read), trim_interval);
        }
    }
}

void BarcodeClassifierNode::barcode(SimplexRead& read) {
    auto barcoding_info = get_barcoding_info(read);
    if (!barcoding_info) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(*barcoding_info);

    // get the sequence to map from the record
    auto bc_res = barcoder->barcode(read.read_common.seq, barcoding_info->barcode_both_ends,
                                    barcoding_info->allowed_barcodes);
    read.read_common.barcode = generate_barcode_string(bc_res);
    read.read_common.barcoding_result = std::make_shared<BarcodeScoreResult>(std::move(bc_res));
    if (barcoding_info->trim) {
        read.read_common.barcode_trim_interval = Trimmer::determine_trim_interval(
                *read.read_common.barcoding_result, int(read.read_common.seq.length()));
        Trimmer::trim_sequence(read, read.read_common.barcode_trim_interval);
    }

    m_num_records++;
}

stats::NamedStats BarcodeClassifierNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_barcodes_demuxed"] = m_num_records.load();
    {
        for (const auto& [bc_name, bc_count] : m_barcode_count) {
            std::string key = "bc." + bc_name;
            stats[key] = static_cast<float>(bc_count);
        }
    }
    return stats;
}

}  // namespace dorado
