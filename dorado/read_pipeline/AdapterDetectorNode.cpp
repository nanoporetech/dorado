#include "AdapterDetectorNode.h"

#include "ClientInfo.h"
#include "demux/AdapterDetector.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"
#include "utils/trim.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

// A Node which encapsulates running adapter and primer detection on each read.
AdapterDetectorNode::AdapterDetectorNode(int threads) : MessageSink(10000, threads) {
    start_input_processing(&AdapterDetectorNode::input_thread_fn, this);
}

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

std::shared_ptr<const demux::AdapterDetector> AdapterDetectorNode::get_detector(
        const demux::AdapterInfo& adapter_info) {
    if (!adapter_info.trim_adapters && !adapter_info.trim_primers) {
        return nullptr;
    }
    return m_detector_selector.get_detector(adapter_info);
}

void AdapterDetectorNode::process_read(BamMessage& bam_message) {
    bam1_t* irecord = bam_message.bam_ptr.get();
    bool is_input_reversed = irecord->core.flag & BAM_FREVERSE;
    std::string seq = utils::extract_sequence(irecord);
    if (is_input_reversed) {
        seq = utils::reverse_complement(seq);
    }
    int seqlen = irecord->core.l_qseq;

    auto increment_read_count = utils::PostCondition([this] { m_num_records++; });

    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    const auto& adapter_info =
            bam_message.client_info->contexts().get_ptr<const demux::AdapterInfo>();
    if (!adapter_info) {
        return;
    }

    auto detector = get_detector(*adapter_info);
    if (adapter_info->trim_adapters) {
        auto adapter_res = detector->find_adapters(seq);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    if (adapter_info->trim_primers) {
        auto primer_res = detector->find_primers(seq);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    }

    if (adapter_info->trim_adapters || adapter_info->trim_primers) {
        std::pair<int, int> trim_interval = adapter_trim_interval;
        trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
        trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
        if (trim_interval.first >= trim_interval.second) {
            spdlog::warn("Unexpected adapter/primer trim interval {}-{} for {}",
                         trim_interval.first, trim_interval.second, seq);
            bam_message.bam_ptr = utils::new_unmapped_record(irecord, {}, {});
            return;
        }
        bam_message.bam_ptr = Trimmer::trim_sequence(irecord, trim_interval);
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
    auto detector = get_detector(*adapter_info);
    if (adapter_info->trim_adapters) {
        auto adapter_res = detector->find_adapters(read.read_common.seq);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    if (adapter_info->trim_primers) {
        auto primer_res = detector->find_primers(read.read_common.seq);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    }
    if (adapter_info->trim_adapters || adapter_info->trim_primers) {
        std::pair<int, int> trim_interval = adapter_trim_interval;
        trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
        trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
        if (trim_interval.first >= trim_interval.second) {
            spdlog::warn("Unexpected adapter/primer trim interval {}-{} for {}",
                         trim_interval.first, trim_interval.second, read.read_common.seq);
            return;
        }
        demux::AdapterDetector::check_and_update_barcoding(read, trim_interval);
        Trimmer::trim_sequence(read, trim_interval);
        read.read_common.adapter_trim_interval = trim_interval;
    }
}

stats::NamedStats AdapterDetectorNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_reads_trimmed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
