#include "AdapterDetectorNode.h"

#include "ClientInfo.h"
#include "demux/AdapterDetector.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "messages.h"
#include "torch_utils/trim.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

// A Node which encapsulates running adapter and primer detection on each read.
AdapterDetectorNode::AdapterDetectorNode(int threads) : MessageSink(10000, threads) {}

void AdapterDetectorNode::input_thread_fn() {
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

std::shared_ptr<demux::AdapterDetector> AdapterDetectorNode::get_detector(
        const demux::AdapterInfo& adapter_info) {
    if (!adapter_info.trim_adapters && !adapter_info.trim_primers) {
        return nullptr;
    }
    return m_detector_selector.get_detector(adapter_info);
}

void AdapterDetectorNode::process_read(BamMessage& bam_message) {
    bam1_t* irecord = bam_message.bam_ptr.get();
    bool is_input_reversed = irecord->core.flag & BAM_FREVERSE;
    std::string qname = bam_get_qname(irecord);
    std::string seq = utils::extract_sequence(irecord);
    if (is_input_reversed) {
        seq = utils::reverse_complement(seq);
    }
    int seqlen = irecord->core.l_qseq;

    const auto& adapter_info =
            bam_message.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    if (!adapter_info) {
        return;
    }

    auto kit_name = bam_message.sequencing_kit;
    if (kit_name.empty()) {
        if (adapter_info->kit_name) {
            // For the standalone trim application, the kit name is provided on
            // the command-line, and passed here via the adapter_info object.
            kit_name = adapter_info->kit_name.value();
        } else {
            // If no kit-name has been provided, then only look for adapters
            // and/or primers that are have been designated as valid targets for
            // any kit. Normally these will only exist if a custom primer file
            // was used.
            kit_name = "ANY";
        }
    }
    // All kit names are uppercase in the AdapterDetector lookup-tables.
    kit_name = utils::to_uppercase(kit_name);

    auto increment_read_count = utils::PostCondition([this] { m_num_records++; });

    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    auto detector = get_detector(*adapter_info);
    if (adapter_info->trim_adapters) {
        auto adapter_res = detector->find_adapters(seq, kit_name);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    AdapterScoreResult primer_res;
    if (adapter_info->trim_primers) {
        primer_res = detector->find_primers(seq, kit_name);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
        bam_message.primer_classification =
                detector->classify_primers(primer_res, primer_trim_interval, seq);
    }
    if (adapter_info->trim_adapters || adapter_info->trim_primers) {
        std::pair<int, int> trim_interval = adapter_trim_interval;
        trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
        trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
        if (trim_interval.first >= trim_interval.second) {
            spdlog::trace(
                    "Adapter and/or primer detected for read {}, but could not be "
                    "trimmed due to short length.",
                    qname);
            ++m_num_untrimmed_short_reads;
            return;
        }
        bam_message.adapter_trim_interval = trim_interval;
    }
}

void AdapterDetectorNode::process_read(SimplexRead& read) {
    // get the sequence to map from the record
    auto seqlen = int(read.read_common.seq.length());

    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    auto increment_read_count = utils::PostCondition([this] { m_num_records++; });

    const auto& adapter_info =
            read.read_common.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    if (!adapter_info) {
        return;
    }

    // If no kit-name has been provided, then only look for adapters and/or
    // primers that are have been designated as valid targets for any kit.
    // Normally these will only exist if a custom primer file was used.
    auto kit_name = read.read_common.sequencing_kit;
    if (kit_name.empty()) {
        kit_name = "ANY";
    }
    // All kit names are uppercase in the AdapterDetector lookup-tables.
    kit_name = utils::to_uppercase(kit_name);

    auto detector = get_detector(*adapter_info);
    if (adapter_info->trim_adapters) {
        auto adapter_res = detector->find_adapters(read.read_common.seq, kit_name);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    AdapterScoreResult primer_res;
    if (adapter_info->trim_primers) {
        primer_res = detector->find_primers(read.read_common.seq, kit_name);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
        read.read_common.primer_classification =
                detector->classify_primers(primer_res, primer_trim_interval, read.read_common.seq);
    }
    if (adapter_info->trim_adapters || adapter_info->trim_primers) {
        std::pair<int, int> trim_interval = adapter_trim_interval;
        trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
        trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
        if (trim_interval.first >= trim_interval.second) {
            spdlog::trace(
                    "Adapter and/or primer detected for read {}, but could not be "
                    "trimmed due to short length.",
                    read.read_common.read_id);
            ++m_num_untrimmed_short_reads;
            return;
        }
        read.read_common.adapter_trim_interval = trim_interval;
    }
}

stats::NamedStats AdapterDetectorNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_reads_processed"] = m_num_records.load();
    stats["num_untrimmed_short_reads"] = m_num_untrimmed_short_reads.load();
    return stats;
}

}  // namespace dorado
