#include "BarcodeClassifierNode.h"

#include "ClientInfo.h"
#include "demux/BarcodeClassifier.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "demux/barcoding_info.h"
#include "messages.h"
#include "torch_utils/trim.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

const std::string UNCLASSIFIED_BARCODE = "unclassified";

std::string generate_barcode_string(const dorado::BarcodeScoreResult& bc_res) {
    std::string bc;
    if (bc_res.barcode_name != UNCLASSIFIED_BARCODE) {
        bc = dorado::barcode_kits::generate_standard_barcode_name(bc_res.kit, bc_res.barcode_name);
    } else {
        bc = UNCLASSIFIED_BARCODE;
    }
    spdlog::trace("BC: {}", bc);
    return bc;
}

const dorado::demux::BarcodingInfo* get_barcoding_info(const dorado::ClientInfo& client_info) {
    auto info = client_info.contexts().get_ptr<const dorado::demux::BarcodingInfo>();
    if (!info || info->kit_name.empty()) {
        return nullptr;
    }
    return info.get();
}

}  // namespace

namespace dorado {

BarcodeClassifierNode::BarcodeClassifierNode(int threads) : MessageSink(10000, threads) {}

void BarcodeClassifierNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamMessage>(message)) {
            auto bam_message = std::get<BamMessage>(std::move(message));
            // If the read is a secondary or supplementary read, ignore it if
            // client requires read trimming.
            const auto* barcoding_info = get_barcoding_info(*bam_message.client_info);
            if (barcoding_info && barcoding_info->trim &&
                (bam_message.bam_ptr->core.flag & (BAM_FSUPPLEMENTARY | BAM_FSECONDARY))) {
                continue;
            }

            barcode(bam_message, barcoding_info);
            send_message_to_sink(std::move(bam_message));
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            barcode(*read);
            send_message_to_sink(std::move(read));
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

void BarcodeClassifierNode::barcode(BamMessage& read, const demux::BarcodingInfo* barcoding_info) {
    if (!barcoding_info) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(*barcoding_info);

    bam1_t* irecord = read.bam_ptr.get();
    bool is_input_reversed = irecord->core.flag & BAM_FREVERSE;
    std::string seq = utils::extract_sequence(irecord);
    if (is_input_reversed) {
        seq = utils::reverse_complement(seq);
    }

    auto bc_res = barcoder->barcode(seq, barcoding_info->barcode_both_ends,
                                    barcoding_info->allowed_barcodes);
    auto bc = generate_barcode_string(bc_res);
    read.barcoding_result = std::make_shared<BarcodeScoreResult>(std::move(bc_res));
    spdlog::trace("Barcode for {} is {}", bam_get_qname(irecord), bc);
    bam_aux_update_str(irecord, "BC", int(bc.length() + 1), bc.c_str());
    m_num_records++;
    {
        std::lock_guard lock(m_barcode_count_mutex);
        m_barcode_count[bc]++;
    }

    int seqlen = irecord->core.l_qseq;
    if (barcoding_info->trim) {
        read.barcode_trim_interval =
                Trimmer::determine_trim_interval(*read.barcoding_result, seqlen);
    }

    auto adapter_primer_info = read.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    bool trim_primers = adapter_primer_info ? adapter_primer_info->trim_primers : false;
    fix_misidentified_primers(*read.barcoding_result, read.adapter_trim_interval, seqlen,
                              trim_primers);
}

void BarcodeClassifierNode::barcode(SimplexRead& read) {
    const auto* barcoding_info = get_barcoding_info(*read.read_common.client_info);
    if (!barcoding_info) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(*barcoding_info);

    // get the sequence to map from the record
    auto bc_res = barcoder->barcode(read.read_common.seq, barcoding_info->barcode_both_ends,
                                    barcoding_info->allowed_barcodes);
    read.read_common.barcode = generate_barcode_string(bc_res);
    spdlog::trace("Barcode for {} is {}", read.read_common.read_id, read.read_common.barcode);
    read.read_common.barcoding_result = std::make_shared<BarcodeScoreResult>(std::move(bc_res));
    int seqlen = int(read.read_common.seq.length());
    if (barcoding_info->trim) {
        read.read_common.barcode_trim_interval =
                Trimmer::determine_trim_interval(*read.read_common.barcoding_result, seqlen);
    }
    auto adapter_primer_info =
            read.read_common.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    bool trim_primers = adapter_primer_info ? adapter_primer_info->trim_primers : false;
    fix_misidentified_primers(*read.read_common.barcoding_result,
                              read.read_common.adapter_trim_interval, seqlen, trim_primers);

    m_num_records++;
    {
        std::lock_guard lock(m_barcode_count_mutex);
        m_barcode_count[read.read_common.barcode]++;
    }
}

void BarcodeClassifierNode::fix_misidentified_primers(const BarcodeScoreResult& bc_res,
                                                      std::pair<int, int>& primer_trim_interval,
                                                      int seqlen,
                                                      bool trim_primers) {
    if (primer_trim_interval.second == 0 ||
        (primer_trim_interval.first == 0 && primer_trim_interval.second == seqlen)) {
        // No adapters or primers were found and/or trimmed, so we have nothing to do here.
        return;
    }
    // Some barcodes may be mistaken for primers during adapter/primer detection. If the trim-interval for adapters/primers
    // overlaps the trim-interval for barcodes, we must assume this is what has happened.
    if (bc_res.top_barcode_pos != std::make_pair(-1, -1)) {
        if (bc_res.top_barcode_pos.first < primer_trim_interval.first) {
            // Overlap detected. If we're trimming primers, then trim up to the beginning of the barcode.
            if (trim_primers) {
                primer_trim_interval.first = bc_res.top_barcode_pos.first;
            } else {
                // If we're not trimming primers, we don't know anymore whether a front adapter was found, or where. So don't trim from the beginning at all.
                primer_trim_interval.first = 0;
            }
        }
    }
    if (bc_res.bottom_barcode_pos != std::make_pair(-1, -1)) {
        if (bc_res.bottom_barcode_pos.second > primer_trim_interval.second) {
            // Overlap detected. If we're trimming primers, then trim from the end of the barcode.
            if (trim_primers) {
                primer_trim_interval.second = bc_res.bottom_barcode_pos.second;
            } else {
                // If we're not trimming primers, we don't know anymore whether a rear adapter was found, or where. So don't trim from the end at all.
                primer_trim_interval.second = seqlen;
            }
        }
    }
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
