#include "RNASplitNode.h"

#include "read_utils.h"
#include "splitter_utils.h"
#include "utils/alignment_utils.h"
#include "utils/duplex_utils.h"
#include "utils/sequence_utils.h"
#include "utils/time_utils.h"
#include "utils/uuid_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <optional>
#include <string>

namespace dorado {

RNASplitNode::ExtRead RNASplitNode::create_ext_read(SimplexReadPtr r) const {
    ExtRead ext_read;
    ext_read.read = std::move(r);
    ext_read.possible_pore_regions =
            detect_pore_signal<int16_t>(ext_read.read->read_common.raw_data, m_settings.pore_thr,
                                        m_settings.pore_cl_dist, m_settings.expect_pore_prefix);
    for (auto range : ext_read.possible_pore_regions) {
        spdlog::trace("Pore range {}-{} {}", range.first, range.second,
                      ext_read.read->read_common.read_id);
    }
    return ext_read;
}

std::vector<SimplexReadPtr> RNASplitNode::subreads(SimplexReadPtr read,
                                                   const std::vector<PosRange>& spacers) const {
    std::vector<SimplexReadPtr> subreads;
    subreads.reserve(spacers.size() + 1);

    if (spacers.empty()) {
        subreads.push_back(std::move(read));
        return subreads;
    }

    uint64_t start_pos = 0;
    for (auto r : spacers) {
        if (start_pos < r.first) {
            subreads.push_back(subread(*read, std::nullopt, {start_pos, r.first}));
        }
        start_pos = r.second;
    }
    if (start_pos < read->read_common.get_raw_data_samples()) {
        subreads.push_back(subread(*read, std::nullopt,
                                   {start_pos, read->read_common.get_raw_data_samples()}));
    }

    return subreads;
}

std::vector<std::pair<std::string, RNASplitNode::SplitFinderF>> RNASplitNode::build_split_finders()
        const {
    std::vector<std::pair<std::string, SplitFinderF>> split_finders;
    split_finders.push_back({"PORE_ADAPTER", [&](const ExtRead& read) {
                                 return filter_ranges(read.possible_pore_regions,
                                                      [&](PosRange r) { return true; });
                             }});

    return split_finders;
}

std::vector<SimplexReadPtr> RNASplitNode::split(SimplexReadPtr init_read) const {
    using namespace std::chrono;

    auto start_ts = high_resolution_clock::now();
    auto read_id = init_read->read_common.read_id;
    spdlog::trace("Processing read {}", read_id);

    std::vector<ExtRead> to_split;
    to_split.push_back(create_ext_read(std::move(init_read)));
    for (const auto& [description, split_f] : m_split_finders) {
        spdlog::trace("Running {}", description);
        std::vector<ExtRead> split_round_result;
        for (auto& r : to_split) {
            auto spacers = split_f(r);
            spdlog::trace("RSN: {} strategy {} splits in read {}", description, spacers.size(),
                          read_id);

            if (spacers.empty()) {
                split_round_result.push_back(std::move(r));
            } else {
                for (auto& sr : subreads(std::move(r.read), spacers)) {
                    split_round_result.push_back(create_ext_read(std::move(sr)));
                }
            }
        }
        to_split = std::move(split_round_result);
    }

    std::vector<SimplexReadPtr> split_result;
    size_t subread_id = 0;
    for (auto& ext_read : to_split) {
        if (!ext_read.read->read_common.parent_read_id.empty()) {
            ext_read.read->read_common.subread_id = subread_id++;
            ext_read.read->read_common.split_count = to_split.size();
            const auto subread_uuid =
                    utils::derive_uuid(ext_read.read->read_common.parent_read_id,
                                       std::to_string(ext_read.read->read_common.subread_id));
            ext_read.read->read_common.read_id = subread_uuid;
        }

        split_result.push_back(std::move(ext_read.read));
    }

    spdlog::trace("Read {} split into {} subreads", read_id, split_result.size());

    auto stop_ts = high_resolution_clock::now();
    spdlog::trace("READ duration: {} microseconds (ID: {})",
                  duration_cast<microseconds>(stop_ts - start_ts).count(), read_id);

    return split_result;
}

void RNASplitNode::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto init_read = std::get<SimplexReadPtr>(std::move(message));
        for (auto& subread : split(std::move(init_read))) {
            //TODO correctly process end_reason when we have them
            send_message_to_sink(std::move(subread));
        }
    }
}

RNASplitNode::RNASplitNode(RNASplitSettings settings, int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads),
          m_settings(std::move(settings)),
          m_num_worker_threads(num_worker_threads) {
    m_split_finders = build_split_finders();
    start_threads();
}

void RNASplitNode::start_threads() {
    for (int i = 0; i < m_num_worker_threads; ++i) {
        m_worker_threads.push_back(
                std::make_unique<std::thread>(&RNASplitNode::worker_thread, this));
    }
}

void RNASplitNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();
}

void RNASplitNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats RNASplitNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
