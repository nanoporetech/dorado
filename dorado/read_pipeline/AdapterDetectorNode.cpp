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

void AdapterDetectorNode::check_and_update_barcoding(SimplexRead& read,
                                                     std::pair<int, int>& trim_interval) {
    // If barcoding has been done, we may need to make some adjustments.
    if (!read.read_common.barcoding_result) {
        return;
    }
    int post_barcode_seq_len = int(read.read_common.pre_trim_seq_length);
    if (read.read_common.barcode_trim_interval.first != 0 &&
        read.read_common.barcode_trim_interval.second != 0) {
        post_barcode_seq_len = read.read_common.barcode_trim_interval.second -
                               read.read_common.barcode_trim_interval.first;
    }
    if (trim_interval.first > 0) {
        // An adapter or primer was found at the beginning of the read.
        // If any barcodes were found, their position details will need to be updated
        // so that they refer to the position in the trimmed read. If the barcode
        // overlaps the region we are planning to trim, then this probably means that
        // the barcode was misidentified as a primer, so we should not trim it.
        if (read.read_common.barcoding_result) {
            auto& barcode_result = *read.read_common.barcoding_result;
            if (barcode_result.barcode_name != "unclassified") {
                if (read.read_common.barcode_trim_interval.first > 0) {
                    // We've already trimmed a front barcode. Adapters and primers do not appear after barcodes, so
                    // we should ignore this.
                    trim_interval.first = 0;
                } else {
                    if (barcode_result.top_barcode_pos != std::pair<int, int>(-1, -1)) {
                        // We have detected, but not trimmed, a front barcode.
                        if (barcode_result.top_barcode_pos.first < trim_interval.first) {
                            // We've misidentified the barcode as a primer. Ignore it.
                            trim_interval.first = 0;
                        } else {
                            // Update the position to correspond to the trimmed sequence.
                            barcode_result.top_barcode_pos.first -= trim_interval.first;
                            barcode_result.top_barcode_pos.second -= trim_interval.first;
                        }
                    }
                    if (barcode_result.bottom_barcode_pos != std::pair<int, int>(-1, -1) &&
                        read.read_common.barcode_trim_interval.second != 0 &&
                        read.read_common.barcode_trim_interval.second !=
                                int(read.read_common.pre_trim_seq_length)) {
                        // We have detected, but not trimmed, a rear barcode.
                        // Update the position to correspond to the trimmed sequence.
                        barcode_result.bottom_barcode_pos.first -= trim_interval.first;
                        barcode_result.bottom_barcode_pos.second -= trim_interval.second;
                    }
                }
            }
        }
    }
    if (trim_interval.second > 0 && trim_interval.second != post_barcode_seq_len) {
        // An adapter or primer was found at the end of the read.
        // This does not require any barcode positions to be updated, but if the
        // barcode overlaps the region we are planning to trim, then this probably
        // means that the barcode was misidentified as a primer, so we should not
        // trim it.
        if (read.read_common.barcoding_result) {
            auto& barcode_result = *read.read_common.barcoding_result;
            if (barcode_result.barcode_name != "unclassified") {
                if (read.read_common.barcode_trim_interval.second > 0 &&
                    read.read_common.barcode_trim_interval.second !=
                            int(read.read_common.pre_trim_seq_length)) {
                    // We've already trimmed a rear barcode. Adapters and primers do not appear before rear barcodes,
                    // so we should ignore this.
                    trim_interval.second = post_barcode_seq_len;
                } else if (barcode_result.bottom_barcode_pos.second > trim_interval.second) {
                    // We've misidentified the rear barcode as a primer. Ignore it.
                    trim_interval.second = post_barcode_seq_len;
                }
            }
        }
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
        check_and_update_barcoding(read, trim_interval);
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
