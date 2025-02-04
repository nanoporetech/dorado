#include "TrimmerNode.h"

#include "ClientInfo.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "demux/barcoding_info.h"
#include "torch_utils/trim.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
using Interval = std::pair<int, int>;
std::tuple<bool, bool, Interval> get_trim_interval(dorado::ClientInfo& client_info,
                                                   int seqlen,
                                                   Interval adapter_interval,
                                                   Interval barcoding_interval,
                                                   const std::string_view read_id) {
    const auto& adapter_info = client_info.contexts().get_ptr<const dorado::demux::AdapterInfo>();
    const auto& barcode_info = client_info.contexts().get_ptr<const dorado::demux::BarcodingInfo>();

    if ((!adapter_info || (!adapter_info->trim_adapters && !adapter_info->trim_primers)) &&
        (!barcode_info || !barcode_info->trim)) {
        // No trimming to be done, and no need to strip alignment information from read.
        return {false, false, {0, 0}};
    }

    // If there is no trimming to be done for this read, then the left trim-point will be zero, and the right trim-point will either be zero or seqlen.
    bool trim_adapter = false, trim_barcodes = false;
    if (adapter_interval.second > 0 &&
        (adapter_interval.second < seqlen || adapter_interval.first > 0)) {
        trim_adapter = true;
    }
    if (barcoding_interval.second > 0 &&
        (barcoding_interval.second < seqlen || barcoding_interval.first > 0)) {
        trim_barcodes = true;
    }

    // Find the inner-most trim locations.
    Interval trim_interval = {0, seqlen};
    if (trim_adapter) {
        trim_interval = adapter_interval;
    }
    if (trim_barcodes) {
        trim_interval.first = std::max(trim_interval.first, barcoding_interval.first);
        trim_interval.second = std::min(trim_interval.second, barcoding_interval.second);
    }

    if (trim_interval.second <= trim_interval.first) {
        spdlog::debug("Invalid trim interval for read id {}: {}-{}. Trimming will be skipped.",
                      read_id, trim_interval.first, trim_interval.second);
        return {false, false, {0, 0}};
    }

    return {trim_barcodes, trim_adapter, trim_interval};
}
}  // namespace

namespace dorado {

// This Node is responsible for trimming adapters, primers, and barcodes.
TrimmerNode::TrimmerNode(int threads) : MessageSink(10000, threads) {}

void TrimmerNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamMessage>(message)) {
            auto bam_message = std::get<BamMessage>(std::move(message));
            // If the read is a secondary or supplementary read, ignore it.
            if (bam_message.bam_ptr->core.flag & (BAM_FSUPPLEMENTARY | BAM_FSECONDARY)) {
                continue;
            }
            process_read(bam_message);
            send_message_to_sink(std::move(bam_message));
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            process_read(*read);
            send_message_to_sink(std::move(read));
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

void TrimmerNode::process_read(BamMessage& bam_message) {
    bam1_t* irecord = bam_message.bam_ptr.get();
    int seqlen = irecord->core.l_qseq;

    auto increment_read_count = utils::PostCondition([this] { m_num_records++; });

    auto [trim_adapter, trim_barcodes, trim_interval] =
            get_trim_interval(*bam_message.client_info, seqlen, bam_message.adapter_trim_interval,
                              bam_message.barcode_trim_interval, bam_get_qname(irecord));

    if (trim_adapter || trim_barcodes) {
        bam_message.bam_ptr = Trimmer::trim_sequence(irecord, trim_interval);
        if (bam_message.primer_classification.orientation != StrandOrientation::UNKNOWN) {
            auto sense_data = uint8_t(to_char(bam_message.primer_classification.orientation));
            bam_aux_append(bam_message.bam_ptr.get(), "TS", 'A', 1, &sense_data);
        }
        if (!bam_message.primer_classification.umi_tag_sequence.empty()) {
            auto len = int(bam_message.primer_classification.umi_tag_sequence.size()) + 1;
            auto data = (const uint8_t*)bam_message.primer_classification.umi_tag_sequence.c_str();
            bam_aux_append(bam_message.bam_ptr.get(), "RX", 'Z', len, data);
        }
    } else {
        // Even if we don't trim this read, we need to strip any alignment details, since the BAM header
        // will not contain any alignment information anymore.
        bam_message.bam_ptr = utils::new_unmapped_record(irecord, {}, {});
        return;
    }
}

void TrimmerNode::process_read(SimplexRead& read) {
    // get the sequence to map from the record
    auto seqlen = int(read.read_common.seq.length());

    auto increment_read_count = utils::PostCondition([this] { m_num_records++; });

    auto [trim_adapter, trim_barcodes, trim_interval] = get_trim_interval(
            *read.read_common.client_info, seqlen, read.read_common.adapter_trim_interval,
            read.read_common.barcode_trim_interval, read.read_common.read_id);

    if (trim_adapter || trim_barcodes) {
        Trimmer::trim_sequence(read, trim_interval);
        Trimmer::check_and_update_barcoding(read);
    }
}

stats::NamedStats TrimmerNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_reads_trimmed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
