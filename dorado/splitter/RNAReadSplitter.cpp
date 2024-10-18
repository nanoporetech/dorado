#include "RNAReadSplitter.h"

#include "read_pipeline/messages.h"
#include "splitter/splitter_utils.h"
#include "utils/uuid_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <sstream>
#include <string>

using namespace dorado::splitter;

namespace dorado::splitter {

RNAReadSplitter::ExtRead RNAReadSplitter::create_ext_read(SimplexReadPtr r) const {
    ExtRead ext_read;
    ext_read.read = std::move(r);
    ext_read.possible_pore_regions =
            detect_pore_signal<int16_t>(ext_read.read->read_common.raw_data, m_settings.pore_thr,
                                        m_settings.pore_cl_dist, m_settings.expect_pore_prefix);
    for (const auto& range : ext_read.possible_pore_regions) {
        spdlog::trace("Pore range {}-{} {}", range.start_sample, range.end_sample,
                      ext_read.read->read_common.read_id);
    }
    return ext_read;
}

std::vector<SimplexReadPtr> RNAReadSplitter::subreads(SimplexReadPtr read,
                                                      const SampleRanges<int16_t>& spacers) const {
    std::vector<SimplexReadPtr> subreads;
    subreads.reserve(spacers.size() + 1);

    if (spacers.empty()) {
        subreads.push_back(std::move(read));
        return subreads;
    }

    uint64_t start_sample = 0;
    for (const auto& r : spacers) {
        if (start_sample < r.start_sample) {
            subreads.push_back(subread(*read, std::nullopt, {start_sample, r.start_sample}));
        }
        start_sample = r.end_sample;
    }
    if (start_sample < read->read_common.get_raw_data_samples()) {
        subreads.push_back(subread(*read, std::nullopt,
                                   {start_sample, read->read_common.get_raw_data_samples()}));
    }

    return subreads;
}

std::vector<std::pair<std::string, RNAReadSplitter::SplitFinderF>>
RNAReadSplitter::build_split_finders() const {
    std::vector<std::pair<std::string, SplitFinderF>> split_finders;
    split_finders.push_back(
            {"PORE_ADAPTER", [&](const ExtRead& read) { return read.possible_pore_regions; }});

    return split_finders;
}

std::vector<SimplexReadPtr> RNAReadSplitter::split(SimplexReadPtr init_read) const {
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
    std::ostringstream log_output;
    for (auto& ext_read : to_split) {
        if (!ext_read.read->read_common.parent_read_id.empty()) {
            ext_read.read->read_common.subread_id = subread_id++;
            ext_read.read->read_common.split_count = to_split.size();
            const auto subread_uuid =
                    utils::derive_uuid(ext_read.read->read_common.parent_read_id,
                                       std::to_string(ext_read.read->read_common.subread_id));
            ext_read.read->read_common.read_id = subread_uuid;
        }

        log_output << ext_read.read->read_common.read_id << " ("
                   << ext_read.read->read_common.split_point << "); ";

        split_result.push_back(std::move(ext_read.read));
    }

    spdlog::trace("Read {} split into {} subreads: {}", read_id, split_result.size(),
                  log_output.str());

    auto stop_ts = high_resolution_clock::now();
    spdlog::trace("READ duration: {} microseconds (ID: {})",
                  duration_cast<microseconds>(stop_ts - start_ts).count(), read_id);

    return split_result;
}

RNAReadSplitter::RNAReadSplitter(RNASplitSettings settings) : m_settings(std::move(settings)) {
    m_split_finders = build_split_finders();
}

}  // namespace dorado::splitter
