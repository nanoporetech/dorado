#include "AdapterDetector.h"

#include "parse_custom_kit.h"
#include "parse_custom_sequences.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <edlib.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

const int ADAPTER_TRIM_LENGTH = 75;
const int PRIMER_TRIM_LENGTH = 150;

// Create edlib configuration for detecting adapters and primers.
EdlibAlignConfig init_edlib_config_for_adapters() {
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_LOC;
    // Currently none of our adapters or primers have Ns, but we should support them.
    static const EdlibEqualityPair additionalEqualities[4] = {
            {'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 4;
    return placement_config;
}

dorado::SingleEndResult copy_results(const EdlibAlignResult& source,
                                     const std::string& name,
                                     size_t length) {
    dorado::SingleEndResult dest{};
    dest.name = name;

    if (source.status != EDLIB_STATUS_OK || !source.startLocations || !source.endLocations) {
        return dest;
    }

    dest.score = 1.0f - float(source.editDistance) / length;
    dest.position = {source.startLocations[0], source.endLocations[0]};
    return dest;
}

void align(std::string_view q,
           std::string_view t,
           const std::string& name,
           std::vector<dorado::SingleEndResult>& results,
           const int rear_start,
           const EdlibAlignConfig& config) {
    auto result = edlibAlign(q.data(), int(q.length()), t.data(), int(t.length()), config);
    results.emplace_back(copy_results(result, name, q.length()));

    if (rear_start >= 0) {
        results.back().position.first += rear_start;
        results.back().position.second += rear_start;
    }

    edlibFreeAlignResult(result);
}

dorado::SingleEndResult get_best_result(const std::vector<dorado::SingleEndResult>& results) {
    int best = -1;
    float best_score = -1.0f;
    constexpr float EPSILON = 0.1f;
    for (size_t i = 0; i < results.size(); ++i) {
        int old_span =
                (best == -1) ? 0 : results[best].position.second - results[best].position.first;
        int new_span = results[i].position.second - results[i].position.first;
        if (results[i].score > best_score + EPSILON) {
            // The current match is clearly better than the previously seen best match.
            best_score = results[i].score;
            best = int(i);
        }
        if (std::abs(results[i].score - best_score) <= EPSILON) {
            // The current match and previously seen best match have nearly equal scores. Pick the longer one.
            if (new_span > old_span) {
                best_score = results[i].score;
                best = int(i);
            }
        }
    }

    if (best != -1) {
        return results[best];
    }

    return {};
}

}  // namespace

namespace dorado {
namespace demux {

AdapterDetector::AdapterDetector(const std::optional<std::string>& custom_primer_file) {
    if (custom_primer_file.has_value()) {
        m_sequence_manager = std::make_unique<adapter_primer_kits::AdapterPrimerManager>(
                custom_primer_file.value());
    } else {
        m_sequence_manager = std::make_unique<adapter_primer_kits::AdapterPrimerManager>();
    }
}

AdapterDetector::~AdapterDetector() = default;

AdapterScoreResult AdapterDetector::find_adapters(const std::string& seq,
                                                  const std::string& kit_name) {
    const auto& adapter_sequences = get_adapter_sequences(kit_name);
    return detect(seq, adapter_sequences, ADAPTER);
}

AdapterScoreResult AdapterDetector::find_primers(const std::string& seq,
                                                 const std::string& kit_name) {
    const auto& primer_sequences = get_primer_sequences(kit_name);
    return detect(seq, primer_sequences, PRIMER);
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_adapter_sequences(
        const std::string& kit_name) {
    auto it = m_adapter_sequences.find(kit_name);
    if (it != m_adapter_sequences.end()) {
        return it->second;
    }
    auto adapters = m_sequence_manager->get_adapters(kit_name);
    auto result = m_adapter_sequences.emplace(kit_name, std::move(adapters));
    return result.first->second;
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_primer_sequences(
        const std::string& kit_name) {
    auto it = m_primer_sequences.find(kit_name);
    if (it != m_primer_sequences.end()) {
        return it->second;
    }
    // With primers, we not only look for the front sequence at the start
    // of the read, and the rear sequence at the end, but we also need to
    // look for the reverse of the rear sequence at the start of the read,
    // and the reverse of the front sequence at the end of the read.
    auto primers = m_sequence_manager->get_primers(kit_name);
    std::vector<Query> primer_queries;
    for (const auto& primer : primers) {
        primer_queries.push_back(
                {primer.name + "_FWD", primer.front_sequence, primer.rear_sequence});
        auto front_rev_seq = utils::reverse_complement(primer.front_sequence);
        auto rear_rev_seq = utils::reverse_complement(primer.rear_sequence);
        primer_queries.push_back({primer.name + "_REV", rear_rev_seq, front_rev_seq});
    }
    auto result = m_primer_sequences.emplace(kit_name, std::move(primer_queries));
    return result.first->second;
}

AdapterScoreResult AdapterDetector::detect(const std::string& seq,
                                           const std::vector<Query>& queries,
                                           AdapterDetector::QueryType query_type) const {
    const std::string_view seq_view(seq);
    const auto TRIM_LENGTH = (query_type == ADAPTER ? ADAPTER_TRIM_LENGTH : PRIMER_TRIM_LENGTH);
    const std::string_view read_front = seq_view.substr(0, TRIM_LENGTH);
    int rear_start = std::max(0, int(seq.length()) - TRIM_LENGTH);
    const std::string_view read_rear = seq_view.substr(rear_start, TRIM_LENGTH);

    // Try to find the location of the queries in the front and rear windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_adapters();

    std::vector<SingleEndResult> front_results, rear_results;
    constexpr int IS_FRONT = -1;
    for (size_t i = 0; i < queries.size(); i++) {
        const auto& name = queries[i].name;
        std::string_view query_seq_front = queries[i].front_sequence;
        std::string_view query_seq_rear = queries[i].rear_sequence;
        spdlog::trace("Checking adapter/primer {}", name);

        if (!query_seq_front.empty()) {
            align(query_seq_front, read_front, name + "_FRONT", front_results, IS_FRONT,
                  placement_config);
        }
        if (!query_seq_rear.empty()) {
            align(query_seq_rear, read_rear, name + "_REAR", rear_results, rear_start,
                  placement_config);
        }
    }
    return {get_best_result(front_results), get_best_result(rear_results)};
}

}  // namespace demux
}  // namespace dorado
