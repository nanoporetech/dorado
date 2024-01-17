#include "AdapterDetectorNode.h"

#include "demux/Trimmer.h"
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

namespace dorado {

// A Node which encapsulates running adapter and primer detection on each read.
AdapterDetectorNode::AdapterDetectorNode(int threads, bool trim_adapters, bool trim_primers)
        : MessageSink(10000),
          m_threads(threads),
          m_trim_adapters(trim_adapters),
          m_trim_primers(trim_primers) {
    start_threads();
}

AdapterDetectorNode::AdapterDetectorNode(int threads)
        : MessageSink(10000), m_threads(threads), m_trim_adapters(true), m_trim_primers(true) {
    start_threads();
}

void AdapterDetectorNode::start_threads() {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(std::make_unique<std::thread>(
                std::thread(&AdapterDetectorNode::worker_thread, this)));
    }
}

void AdapterDetectorNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
    m_workers.clear();
}

void AdapterDetectorNode::restart() {
    restart_input_queue();
    start_threads();
}

AdapterDetectorNode::~AdapterDetectorNode() { terminate_impl(); }

void AdapterDetectorNode::worker_thread() {
    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            process_read(read);
            send_message_to_sink(std::move(read));
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            process_read(*read);
            send_message_to_sink(std::move(read));
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

void AdapterDetectorNode::process_read(BamPtr& read) {
    bam1_t* irecord = read.get();
    std::string seq = utils::extract_sequence(irecord);
    int seqlen = irecord->core.l_qseq;

    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};
    if (m_trim_adapters) {
        auto adapter_res = m_detector.find_adapters(seq);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    if (m_trim_primers) {
        auto primer_res = m_detector.find_primers(seq);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    }
    m_num_records++;

    if (m_trim_adapters || m_trim_primers) {
        std::pair<int, int> trim_interval = adapter_trim_interval;
        trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
        trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
        if (trim_interval.first >= trim_interval.second) {
            spdlog::warn("Unexpected adapter/primer trim interval {}-{} for {}",
                         trim_interval.first, trim_interval.second, seq);
            return;
        }
        read = Trimmer::trim_sequence(std::move(read), trim_interval);
    }
}

void AdapterDetectorNode::process_read(SimplexRead& read) {
    // get the sequence to map from the record
    auto seqlen = int(read.read_common.seq.length());

    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    // Check read for instruction on what to trim.
    bool trim_adapters = m_trim_adapters;
    bool trim_primers = m_trim_primers;
    if (read.read_common.adapter_info) {
        // The read contains instruction on what to trim, so ignore class defaults.
        trim_adapters = read.read_common.adapter_info->trim_adapters;
        trim_primers = read.read_common.adapter_info->trim_primers;
    }
    if (trim_adapters) {
        auto adapter_res = m_detector.find_adapters(read.read_common.seq);
        adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    }
    if (trim_primers) {
        auto primer_res = m_detector.find_primers(read.read_common.seq);
        primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    }
    if (trim_adapters || trim_primers) {
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
    m_num_records++;
}

stats::NamedStats AdapterDetectorNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_reads_trimmed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
