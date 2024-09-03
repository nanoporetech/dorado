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

    const auto& adapter_info =
            bam_message.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    const auto& barcode_info =
            bam_message.client_info->contexts().get_ptr<const dorado::demux::BarcodingInfo>();

    if ((!adapter_info || (!adapter_info->trim_adapters && !adapter_info->trim_primers)) &&
        (!barcode_info || !barcode_info->trim)) {
        // No trimming to be done, and no need to strip alignment information from read.
        return;
    }

    // If there is no trimming to be done for this read, then the left trim-point will be zero, and the right trim-point will either be zero or seqlen.
    bool trim_adapter = false, trim_barcodes = false;
    if (bam_message.adapter_trim_interval.second > 0 &&
        (bam_message.adapter_trim_interval.second < seqlen ||
         bam_message.adapter_trim_interval.first > 0)) {
        trim_adapter = true;
    }
    if (bam_message.barcode_trim_interval.second > 0 &&
        (bam_message.barcode_trim_interval.second < seqlen ||
         bam_message.barcode_trim_interval.first > 0)) {
        trim_barcodes = true;
    }

    // Find the inner-most trim locations.
    std::pair<int, int> trim_interval = {0, seqlen};
    if (trim_adapter) {
        trim_interval = bam_message.adapter_trim_interval;
    }
    if (trim_barcodes) {
        trim_interval.first =
                std::max(trim_interval.first, bam_message.barcode_trim_interval.first);
        trim_interval.second =
                std::min(trim_interval.second, bam_message.barcode_trim_interval.second);
    }
    if (trim_adapter || trim_barcodes) {
        bam_message.bam_ptr = Trimmer::trim_sequence(irecord, trim_interval);
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

    const auto& adapter_info =
            read.read_common.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    const auto& barcode_info =
            read.read_common.client_info->contexts().get_ptr<const dorado::demux::BarcodingInfo>();

    if ((!adapter_info || (!adapter_info->trim_adapters && !adapter_info->trim_primers)) &&
        (!barcode_info || !barcode_info->trim)) {
        // No trimming to be done, and no need to strip alignment information from read.
        return;
    }

    // If there is no trimming to be done for this read, then the left trim-point will be zero, and the right trim-point will either be zero or seqlen.
    bool trim_adapter = false, trim_barcodes = false;
    if (read.read_common.adapter_trim_interval.second > 0 &&
        (read.read_common.adapter_trim_interval.second < seqlen ||
         read.read_common.adapter_trim_interval.first > 0)) {
        trim_adapter = true;
    }
    if (read.read_common.barcode_trim_interval.second > 0 &&
        (read.read_common.barcode_trim_interval.second < seqlen ||
         read.read_common.barcode_trim_interval.first > 0)) {
        trim_barcodes = true;
    }

    // Find the inner-most trim locations.
    std::pair<int, int> trim_interval = {0, seqlen};
    if (trim_adapter) {
        trim_interval = read.read_common.adapter_trim_interval;
    }
    if (trim_barcodes) {
        trim_interval.first =
                std::max(trim_interval.first, read.read_common.barcode_trim_interval.first);
        trim_interval.second =
                std::min(trim_interval.second, read.read_common.barcode_trim_interval.second);
    }
    read.read_common.adapter_trim_interval = trim_interval;
    if (trim_adapter || trim_barcodes) {
        Trimmer::trim_sequence(read, trim_interval);
    }

    Trimmer::check_and_update_barcoding(read);
}

stats::NamedStats TrimmerNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_reads_trimmed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
