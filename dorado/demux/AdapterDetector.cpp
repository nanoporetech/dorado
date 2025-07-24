#include "demux/AdapterDetector.h"

#include "adapter_primer_kits.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "utils/log_utils.h"
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

namespace dorado::demux {

namespace {

// The PCS114 and PCB114_24 kits can include UMI tag sequences.
// When present, the tag will either immediately follow the PCS110 SSP sequence near the beginning
// of the read, or its RC will immediately precede the RC of the PCS110 SSP sequence near the end
// of the read. Note that the Vs are wildcards, which could be any of "A", "C", or "G".
const std::string umi_search_pattern = "TTTVVVVTTVVVVTTVVVVTTVVVVTTT";

// For the GEN10X special primers, depending on the specific configuration, there may be a TSO
// sequence following the UMI tag (if not, there will be a run of polyT). And for the VNP sequence,
// there may be an additional bit of sequence just before the VNP sequence. This is because there
// are actually two different VNP primers, but they are nearly identical, except for this extra
// bit of sequence, so treating them as different primers would greatly complicate things.
const std::string gen10x_tso_sequence = "TTTCTTATATGGG";
const std::string gen10x_polyt_sequence = "TTTTTTTTTTTTT";

// This indicates how many bases before the end of the detected SSP primer the UMI search window
// should begin. Note that the first 3 bases of the UMI tag are also the last 3 bases of the primer.
constexpr int UMI_WINDOW_FRONT_OVERLAP = 6;

// The total length of the window used to search for the UMI tag.
constexpr int UMI_WINDOW_LENGTH = 40;

constexpr int ADAPTER_TRIM_LENGTH = 75;
constexpr int PRIMER_TRIM_LENGTH = 150;
constexpr float UMI_SCORE_THRESHOLD = 0.8f;

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

// Create edlib configuration for detecting umi tags.
EdlibAlignConfig init_edlib_config_for_umi_tags() {
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_LOC;
    static const EdlibEqualityPair additionalEqualities[3] = {{'V', 'A'}, {'V', 'C'}, {'V', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 3;
    return placement_config;
}

dorado::SingleEndResult copy_results(const EdlibAlignResult& source, size_t length) {
    dorado::SingleEndResult dest{};

    if (source.status != EDLIB_STATUS_OK || !source.startLocations || !source.endLocations) {
        return dest;
    }

    dest.score = 1.0f - float(source.editDistance) / length;
    dest.position = {source.startLocations[0], source.endLocations[0]};
    return dest;
}

dorado::SingleEndResult align(std::string_view q,
                              std::string_view t,
                              const int rear_start,
                              const EdlibAlignConfig& config) {
    auto result = edlibAlign(q.data(), int(q.length()), t.data(), int(t.length()), config);
    dorado::SingleEndResult se_result(copy_results(result, q.length()));

    if (rear_start >= 0) {
        se_result.position.first += rear_start;
        se_result.position.second += rear_start;
    }
    edlibFreeAlignResult(result);
    return se_result;
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
                                                 const std::string& kit_name,
                                                 PrimerAux primer_aux) {
    const auto& primer_sequences = get_primer_sequences(kit_name, primer_aux);
    return detect(seq, primer_sequences, PRIMER);
}

SingleEndResult AdapterDetector::find_umi_tag(const std::string& seq) {
    // This function assumes that you have reverse-complemented the sequence
    // if you are looking for the UMI tag at the end of the read. The passed
    // sequence should be just the bit of the read you expect to find the
    // tag in.
    EdlibAlignConfig placement_config = init_edlib_config_for_umi_tags();
    auto result = align(umi_search_pattern, seq, -1, placement_config);
    if (result.score > 0.f) {
        auto umi_start = result.position.first;
        auto umi_len = result.position.second - umi_start + 1;
        result.name = seq.substr(umi_start, umi_len);
    }
    return result;
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_adapter_sequences(
        const std::string& kit_name) {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_adapter_sequences.find(kit_name);
    if (it != m_adapter_sequences.end()) {
        return it->second;
    }
    auto adapters = m_sequence_manager->get_adapters(kit_name);
    auto result = m_adapter_sequences.emplace(kit_name, std::move(adapters));
    return result.first->second;
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_primer_sequences(
        const std::string& kit_name,
        PrimerAux primer_aux) {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_primer_sequences.find(kit_name);
    if (it != m_primer_sequences.end()) {
        return it->second;
    }
    // With primers, we look for the front sequence near the start of the read,
    // and the RC of the rear sequence near the end. For reverse reads we need to
    // look for the rear sequence near the beginning of the read, and the RC of
    // the front sequence near the end.
    auto primers = m_sequence_manager->get_primers(kit_name, primer_aux);
    std::vector<Query> primer_queries;
    for (const auto& primer : primers) {
        auto front_rev_seq = utils::reverse_complement(primer.front_sequence);
        auto rear_rev_seq = utils::reverse_complement(primer.rear_sequence);
        primer_queries.push_back({primer.name + "_FWD", primer.front_sequence, rear_rev_seq});
        primer_queries.push_back({primer.name + "_REV", primer.rear_sequence, front_rev_seq});
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
        utils::trace_log("Checking adapter/primer {}", name);

        if (!query_seq_front.empty()) {
            auto result = align(query_seq_front, read_front, IS_FRONT, placement_config);
            result.name = name + "_FRONT";
            front_results.emplace_back(std::move(result));
        }
        if (!query_seq_rear.empty()) {
            auto result = align(query_seq_rear, read_rear, rear_start, placement_config);
            result.name = name + "_REAR";
            rear_results.emplace_back(std::move(result));
        }
    }
    return {get_best_result(front_results), get_best_result(rear_results)};
}

PrimerClassification AdapterDetector::classify_primers(const AdapterScoreResult& result,
                                                       std::pair<int, int>& trim_interval,
                                                       const std::string& sequence) {
    PrimerClassification classification;
    auto strip_suffix = [](const std::string& name, const std::string& suffix) {
        auto k = name.find(suffix);
        if (k != std::string::npos) {
            return name.substr(0, k);
        }
        return name;
    };
    const auto front_name = strip_suffix(result.front.name, "_FRONT");
    const auto rear_name = strip_suffix(result.rear.name, "_REAR");
    auto get_dir = [](const std::string& name) {
        auto k = name.find("_FWD");
        if (k != std::string::npos) {
            return StrandOrientation::FORWARD;
        }
        k = name.find("_REV");
        if (k != std::string::npos) {
            return StrandOrientation::REVERSE;
        }
        return StrandOrientation::UNKNOWN;
    };

    if (front_name != UNCLASSIFIED && rear_name != UNCLASSIFIED) {
        if (front_name == rear_name) {
            classification.primer_name = front_name;
            classification.orientation = get_dir(front_name);
        }
    } else if (front_name != UNCLASSIFIED) {
        classification.primer_name = front_name;
        classification.orientation = get_dir(front_name);
    } else if (rear_name != UNCLASSIFIED) {
        classification.primer_name = rear_name;
        classification.orientation = get_dir(rear_name);
    }
    // For the PCS110 primers, we need to check for a UMI tag after the SSP primer.
    if (classification.primer_name.substr(0, 6) == "PCS110") {
        check_for_umi_tags(result, classification, sequence, trim_interval);
    }
    // For the GEN10X primers, we need to apply some additional analysis checks.
    if (classification.primer_name.substr(0, 3) == "10X") {
        check_10x_primers(classification, sequence, trim_interval);
    }
    return classification;
}

void AdapterDetector::check_for_umi_tags(const AdapterScoreResult& primer_results,
                                         PrimerClassification& classification,
                                         const std::string& sequence,
                                         std::pair<int, int>& trim_interval) {
    std::string search_window;
    if (classification.orientation == StrandOrientation::FORWARD &&
        primer_results.front.name != UNCLASSIFIED) {
        auto a = trim_interval.first - UMI_WINDOW_FRONT_OVERLAP;
        auto b = a + UMI_WINDOW_LENGTH;
        if (a >= 0 && b < int(sequence.size())) {
            search_window = sequence.substr(a, UMI_WINDOW_LENGTH);
        }
    } else if (classification.orientation == StrandOrientation::REVERSE &&
               primer_results.rear.name != UNCLASSIFIED) {
        // We will search for the UMI pattern within the RC of the search window.
        auto b = trim_interval.second + UMI_WINDOW_FRONT_OVERLAP;
        auto a = b - UMI_WINDOW_LENGTH;
        if (a >= 0 && b < int(sequence.size())) {
            search_window = utils::reverse_complement(sequence.substr(a, UMI_WINDOW_LENGTH));
        }
    }
    if (search_window.empty()) {
        return;
    }
    auto result = find_umi_tag(search_window);
    if (result.name.empty() || result.score < UMI_SCORE_THRESHOLD) {
        return;
    }
    // Regardless of strand orientation, we return the UMI sequence as if it were a forward read.
    classification.umi_tag_sequence = result.name;
    // We need to update the trim interval so that the UMI sequence is trimmed.
    if (classification.orientation == StrandOrientation::FORWARD) {
        auto new_pos = trim_interval.first - UMI_WINDOW_FRONT_OVERLAP + result.position.second + 1;
        trim_interval.first = new_pos;
    } else if (classification.orientation == StrandOrientation::REVERSE) {
        auto new_pos = trim_interval.second + UMI_WINDOW_FRONT_OVERLAP - result.position.second - 1;
        trim_interval.second = new_pos;
    }
}

void AdapterDetector::check_10x_primers(PrimerClassification& classification,
                                        const std::string& sequence,
                                        std::pair<int, int>& trim_interval) {
    // We need to look for the UMI tag after the SSP primer.
    std::string search_window;
    const int UMI_PADDING = 56;
    if (classification.orientation == StrandOrientation::FORWARD && trim_interval.first != 0) {
        int window_start = trim_interval.first;
        int window_end = std::min(trim_interval.second, window_start + UMI_PADDING);
        int window_len = window_end - window_start;
        search_window = sequence.substr(window_start, window_len);
    } else if (classification.orientation == StrandOrientation::REVERSE &&
               trim_interval.second != int(sequence.size())) {
        int window_start = std::max(trim_interval.first, trim_interval.second - UMI_PADDING);
        int window_end = trim_interval.second;
        int window_len = window_end - window_start;
        search_window = utils::reverse_complement(sequence.substr(window_start, window_len));
    }
    if (search_window.empty()) {
        return;
    }
    // Search for both the TSO and PolyT sequences within the window.
    EdlibAlignConfig placement_config = init_edlib_config_for_adapters();
    auto result_polyt = align(gen10x_polyt_sequence, search_window, -1, placement_config);
    auto result_tso = align(gen10x_tso_sequence, search_window, -1, placement_config);
    bool polyt_is_better = (result_polyt.score > result_tso.score);
    const auto& result = polyt_is_better ? result_polyt : result_tso;
    if (result.score > UMI_SCORE_THRESHOLD) {
        // Adjust the trim_window. We trim the TSO sequence, but not the polyT.
        auto trim_point = polyt_is_better ? result.position.first : result.position.second + 1;
        if (classification.orientation == StrandOrientation::FORWARD) {
            trim_interval.first += trim_point;
        } else {
            trim_interval.second -= trim_point;
        }
        // Extract everything betweem the SSP primer and the TSO/PolyT.
        // This is the cell-barcode(s) and the UMI tag. Since we don't know what any of those
        // sequences are, or exactly how long any of them should be, we can't do any further
        // processing on them. So just return the entire chunk of sequence in the RX BAM tag.
        if (trim_point > 0) {
            classification.umi_tag_sequence = search_window.substr(0, result.position.first);
        }
    }
}

}  // namespace dorado::demux
