#include "read_pipeline/nodes/BarcodeClassifierNode.h"

#include "demux/BarcodeClassifier.h"
#include "demux/Trimmer.h"
#include "demux/barcoding_info.h"
#include "hts_utils/bam_utils.h"
#include "read_pipeline/base/ClientInfo.h"
#include "read_pipeline/base/messages.h"
#include "utils/barcode_kits.h"
#include "utils/context_container.h"
#include "utils/log_utils.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace {

constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};

const std::string UNCLASSIFIED_BARCODE = "unclassified";

std::string generate_barcode_string(const dorado::BarcodeScoreResult& bc_res) {
    std::string bc;
    if (bc_res.barcode_name != UNCLASSIFIED_BARCODE) {
        bc = dorado::barcode_kits::generate_standard_barcode_name(bc_res.kit, bc_res.barcode_name);
    } else {
        bc = UNCLASSIFIED_BARCODE;
    }
    dorado::utils::trace_log("BC: {}", bc);
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

BarcodeClassifierNode::BarcodeClassifierNode(
        std::shared_ptr<utils::concurrency::MultiQueueThreadPool> thread_pool,
        utils::concurrency::TaskPriority pipeline_priority)
        : MessageSink(10000, 1),
          m_thread_pool(std::move(thread_pool)),
          m_task_executor(*m_thread_pool, pipeline_priority, MAX_PROCESSING_QUEUE_SIZE) {}

BarcodeClassifierNode::BarcodeClassifierNode(int threads)
        : BarcodeClassifierNode(
                  std::make_shared<utils::concurrency::MultiQueueThreadPool>(threads,
                                                                             "barcode_pool"),
                  utils::concurrency::TaskPriority::normal) {}

BarcodeClassifierNode::~BarcodeClassifierNode() {
    stop_input_processing(utils::AsyncQueueTerminateFast::Yes);
}

std::string BarcodeClassifierNode::get_name() const { return "BarcodeClassifierNode"; }

void BarcodeClassifierNode::terminate(const TerminateOptions& terminate_options) {
    stop_input_processing(terminate_options.fast);
    m_task_executor.flush();
}

void BarcodeClassifierNode::restart() {
    m_task_executor.restart();
    start_input_processing([this] { input_thread_fn(); }, "brcd_classifier");
}

void BarcodeClassifierNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        std::visit(
                [this](auto&& read) {
                    using T = std::decay_t<decltype(read)>;
                    if constexpr (std::is_same_v<T, BamMessage>) {
                        // If the read is a secondary or supplementary read, ignore it if
                        // client requires read trimming.
                        m_task_executor.send([this, read_ = std::move(read)]() mutable {
                            const auto* barcoding_info = get_barcoding_info(*read_.client_info);
                            if (barcoding_info && barcoding_info->trim &&
                                (read_.data->bam_ptr->core.flag &
                                 (BAM_FSUPPLEMENTARY | BAM_FSECONDARY))) {
                                return;  // n.b. discards the read!
                            }

                            barcode(read_, barcoding_info);
                            send_message_to_sink(std::move(read_));
                        });

                    } else if constexpr (std::is_same_v<T, SimplexReadPtr>) {
                        m_task_executor.send([this, read_ = std::move(read)]() mutable {
                            barcode(*read_);
                            send_message_to_sink(std::move(read_));
                        });
                    } else {
                        send_message_to_sink(std::move(read));
                    }
                },
                message);
    }
}

void BarcodeClassifierNode::barcode(BamMessage& message,
                                    const demux::BarcodingInfo* barcoding_info) {
    if (!barcoding_info) {
        return;
    }
    auto barcoder = m_barcoder_selector.get_barcoder(*barcoding_info);

    HtsData& read = *message.data;
    bam1_t* irecord = read.bam_ptr.get();
    bool is_input_reversed = irecord->core.flag & BAM_FREVERSE;
    std::string seq = utils::extract_sequence(irecord);
    if (is_input_reversed) {
        seq = utils::reverse_complement(seq);
    }

    auto bc_res = barcoder->barcode(seq, barcoding_info->barcode_both_ends,
                                    barcoding_info->allowed_barcodes);
    auto bc = generate_barcode_string(bc_res);
    if (barcoding_info->sample_sheet) {
        bc_res.alias = barcoding_info->sample_sheet->get_alias(bc);
        bc_res.type = barcoding_info->sample_sheet->get_sample_type(bc);
        if (!bc_res.alias.empty()) {
            bc = bc_res.alias;
        }
    }

    read.barcoding_result = std::make_shared<BarcodeScoreResult>(std::move(bc_res));
    read.read_attrs.barcode_id =
            barcode_kits::normalize_barcode_name(read.barcoding_result->barcode_name);
    read.read_attrs.barcode_alias = read.barcoding_result->alias;
    utils::trace_log("Barcode for {} is {}", bam_get_qname(irecord), bc);
    if (bc != UNCLASSIFIED_BARCODE) {
        bam_aux_update_str(irecord, "BC", int(bc.length() + 1), bc.c_str());
        bam_aux_update_str(irecord, "bv", int(read.barcoding_result->variant.length() + 1),
                           read.barcoding_result->variant.c_str());

        std::vector<float> barcode_info;
        barcode_info.reserve(7);  // update if dual-barcoding implemented
        barcode_info.push_back(read.barcoding_result->barcode_score);
        barcode_info.push_back(read.barcoding_result->top_barcode_pos.first);  // front_start_index
        barcode_info.push_back(read.barcoding_result->top_barcode_pos.second -
                               read.barcoding_result->top_barcode_pos.first);
        barcode_info.push_back(read.barcoding_result->top_barcode_score);
        barcode_info.push_back(read.barcoding_result->bottom_barcode_pos.second);  // rear_end_index
        barcode_info.push_back(read.barcoding_result->bottom_barcode_pos.second -
                               read.barcoding_result->bottom_barcode_pos.first);
        barcode_info.push_back(read.barcoding_result->bottom_barcode_score);
        bam_aux_update_array(irecord, "bi", 'f', barcode_info.size(), barcode_info.data());

        auto rg_tag = bam_aux_get(irecord, "RG");
        if (rg_tag) {
            std::string rg_tag_value = bam_aux2Z(rg_tag);
            if (!rg_tag_value.ends_with(bc)) {
                if (!rg_tag_value.empty()) {
                    rg_tag_value.append("_");
                }
                rg_tag_value.append(bc);
                bam_aux_update_str(irecord, "RG", int(rg_tag_value.length() + 1),
                                   rg_tag_value.c_str());
            }
        }
    } else {
        auto delete_tag = [irecord](const char* tag) {
            auto aux_tag = bam_aux_get(irecord, tag);
            if (aux_tag) {
                bam_aux_del(irecord, aux_tag);
            }
        };
        delete_tag("BC");
        delete_tag("bv");
        delete_tag("bi");
    }
    m_num_records++;
    if (read.barcoding_result->found_midstrand) {
        m_mid_strand_count++;
    }
    {
        std::lock_guard lock(m_barcode_count_mutex);
        m_barcode_count[bc]++;
    }

    int seqlen = irecord->core.l_qseq;
    if (barcoding_info->trim) {
        read.barcode_trim_interval =
                Trimmer::determine_trim_interval(*read.barcoding_result, seqlen);
    }
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
    utils::trace_log("Barcode for {} is {}", read.read_common.read_id, read.read_common.barcode);
    {
        std::lock_guard lock(m_barcode_count_mutex);
        m_barcode_count[read.read_common.barcode]++;
    }
    if (bc_res.found_midstrand) {
        m_mid_strand_count++;
    }

    if (barcoding_info->sample_sheet) {
        bc_res.alias = barcoding_info->sample_sheet->get_alias(
                read.read_common.flowcell_id, read.read_common.position_id,
                read.read_common.experiment_id, read.read_common.barcode);
        bc_res.type = barcoding_info->sample_sheet->get_sample_type(
                read.read_common.flowcell_id, read.read_common.position_id,
                read.read_common.experiment_id, read.read_common.barcode);
        read.read_common.barcode = bc_res.alias;
    }

    read.read_common.barcoding_result = std::make_shared<BarcodeScoreResult>(std::move(bc_res));
    int seqlen = int(read.read_common.seq.length());
    if (barcoding_info->trim) {
        read.read_common.barcode_trim_interval =
                Trimmer::determine_trim_interval(*read.read_common.barcoding_result, seqlen);
    }
    m_num_records++;
}

stats::NamedStats BarcodeClassifierNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["queued_tasks"] = double(m_task_executor.num_tasks_in_flight());
    stats["num_barcodes_demuxed"] = m_num_records.load();
    {
        std::lock_guard lock(m_barcode_count_mutex);
        for (const auto& [bc_name, bc_count] : m_barcode_count) {
            std::string key = "bc." + bc_name;
            stats[key] = static_cast<float>(bc_count);
        }
    }
    stats["num_midstrand_barcodes"] = m_mid_strand_count.load();

    return stats;
}

}  // namespace dorado
