#include "BarcodeClassifierNode.h"

#include "BarcodeClassifier.h"
#include "htslib/sam.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

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
                                             bool barcode_both_ends)
        : MessageSink(10000), m_threads(threads), m_barcoder(kit_names, barcode_both_ends) {
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
            barcode(read.get());
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

void BarcodeClassifierNode::barcode(bam1_t* irecord) {
    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;
    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);

    auto bc_res = m_barcoder.barcode(seq);
    auto bc = generate_barcode_string(bc_res);
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    m_num_records++;
}

void BarcodeClassifierNode::barcode(ReadPtr& read) {
    // get the sequence to map from the record
    auto bc_res = m_barcoder.barcode(read->seq);
    auto bc = generate_barcode_string(bc_res);
    read->barcode = bc;
    m_num_records++;
}

stats::NamedStats BarcodeClassifierNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_barcodes_demuxed"] = m_num_records.load();
    return stats;
}

}  // namespace dorado
