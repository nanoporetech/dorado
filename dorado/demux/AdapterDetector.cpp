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

// For adapters, we there are specific sequences we look for at the front of the read. We don't look for exactly
// the reverse complement at the rear of the read, though, because it will generally be truncated. So we list here
// the specific sequences to look for at the front and rear of the reads.
// RNA adapters are only looked for at the rear of the read, so don't have a front sequence.
struct Adapter {
    std::string name;
    std::string front_sequence;
    std::string rear_sequence;
};

enum class AdapterCode { LSK109, LSK110, RNA004 };

const std::unordered_map<AdapterCode, Adapter> adapters = {
        {LSK109, {"LSK109", "AATGTACTTCGTTCAGTTACGTATTGCT", "AGCAATACGTAACTGAACGAAGT"}},
        {LSK110, {"LSK110", "CCTGTACTTCGTTCAGTTACGTATTGC", "AGCAATACGTAACTGAAC"}},
        {RNA004, {"RNA004", "", "GGTTGTTTCTGTTGGTGCTG"}}};

const std::unordered_map<AdapterCode, std::set<dorado::models::KitCode>> adapter_kit_map = {
        {LSK109, {SQK_CS9109, SQK_DCS109, SQK_LSK109, SQK_LSK109_XL, SQK_PCS109}},
        {LSK110,
         {SQK_APK114, SQK_LSK110, SQK_LSK110_XL, SQK_LSK111, SQK_LSK111_XL, SQK_LSK112,
          SQK_LSK112_XL, SQK_LSK114, SQK_LSK114_260, SQK_LSK114_XL, SQK_LSK114_XL_260, SQK_PCS111,
          SQK_PCS114, SQK_PCS114_260, SQK_RAD112, SQK_RAD114, SQK_RAD114_260, SQK_ULK114,
          SQK_ULK114_260}},
        {RNA004, {SQK_RNA004}}};

// For primers, we look for each primer sequence, and its reverse complement, at both the front and rear of the read.
struct Primer {
    std::string name;
    std::string sequence;
};

enum class PrimerCode {
    PCR_PSK_rev1,
    PCR_PSK_rev2,
    cDNA_VNP,
    cDNA_SSP,
    PCS110_forward,
    PCS110_reverse,
    RAD
};

const std::unordered_map<PrimerCode, Primer> primers = {
        {PCR_PSK_rev1, {"PCR_PSK_rev1", "ACTTGCCTGTCGCTCTATCTTCGGCGTCTGCTTGGGTGTTTAACC"}},
        {PCR_PSK_rev2, {"PCR_PSK_rev2", "TTTCTGTTGGTGCTGATATTGCGGCGTCTGCTTGGGTGTTTAACCT"}},
        {cDNA_VNP, {"cDNA_VNP", "ACTTGCCTGTCGCTCTATCTTC"}},
        {cDNA_SSP, {"cDNA_SSP", "TTTCTGTTGGTGCTGATATTGCTGGG"}},
        {PCS110_forward,
         {"PCS110_forward",
          "TCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTT"
          "T"}},
        {PCS110_reverse,
         {"PCS110_reverse", "ATCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGTTTCTGTTGGTGCTGATATTGCTTT"}},
        {RAD, {"RAD", "GCTTGGGTGTTTAACCGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA"}}};

const std::unordered_map<PrimerCode, std::set<dorado::models::KitCode>> primer_kit_map = {

};

std::vector<Adapter> get_adapters_for_kit(const std::string& kit_name) {
    auto kit_code = dorado::models::kit_code(kit_name);
    for (const auto& entry : adapter_kit_map) {
        if (entry.second.find(kit_code) != entry.second.end()) {
            return {adapters.at(entry.first)};
        }
    }
    // kit not found in map, so return all known adapters.
    std::vector<Adapter> all_adapters;
    for (const auto& entry : adapters) {
        all_adapters.push_back(entry.second);
    }
    return all_adapters;
}

std::vector<Primer> get_primers_for_kit(const std::string& kit_name) {
    std::vector<Primer> primer_matches;
    auto kit_code = dorado::models::kit_code(kit_name);
    for (const auto& entry : primer_kit_map) {
        if (entry.second.find(kit_code) != entry.second.end()) {
            primer_matches.push_back(primers.at(entry.first));
        }
    }
    // kit not found in map, so return all known primers.
    if (primer_matches.empty()) {
        for (const auto& entry : primers) {
            primer_matches.push_back(entry.second);
        }
    }
    return primer_matches;
}

dorado::demux::AdapterDetector::Query make_adapter_query(const Adapter& adapter) {
    return {adapter.name, adapter.front_sequence, adapter.rear_sequence};
}

dorado::demux::AdapterDetector::Query make_primer_query(const Primer& primer) {
    return {primer.name, primer.sequence, dorado::utils::reverse_complement(primer.sequence)};
}

}  // namespace

namespace dorado {
namespace demux {

AdapterDetector::AdapterDetector(const std::optional<std::string>& custom_primer_file) {
    if (custom_primer_file.has_value()) {
        parse_custom_sequence_file(custom_primer_file.value());
    }
}

AdapterDetector::~AdapterDetector() = default;

void AdapterDetector::parse_custom_sequence_file(const std::string& filename) {
    auto sequence_map = parse_custom_sequences(filename);
    for (const auto& item : sequence_map) {
        Query entry = {item.first, item.second, utils::reverse_complement(item.second)};
        m_custom_primer_sequences.emplace_back(std::move(entry));
    }
    // For testing purposes, we want to make sure the sequences are in a deterministic order.
    // Note that parse_custom_sequences() returns an unordered_map, so the order may vary by
    // platform or implementation.
    std::sort(m_custom_primer_sequences.begin(), m_custom_primer_sequences.end());
}

AdapterScoreResult AdapterDetector::find_adapters(const std::string& seq,
                                                  const std::string& kit_name) {
    const auto& adapter_sequences = get_adapter_sequences(kit_name);
    return detect(seq, adapter_sequences, ADAPTER);
}

AdapterScoreResult AdapterDetector::find_primers(const std::string& seq,
                                                 const std::string& kit_name) {
    if (!m_custom_primer_sequences.empty()) {
        return detect(seq, m_custom_primer_sequences, PRIMER);
    }
    const auto& primer_sequences = get_primer_sequences(kit_name);
    return detect(seq, primer_sequences, PRIMER);
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_adapter_sequences(
        const std::string& kit_name) {
    auto it = m_adapter_sequences.find(kit_name);
    if (it != m_adapter_sequences.end()) {
        return it->second;
    }
    const auto& adapters = get_adapters_for_kit(kit_name);
    std::vector<AdapterDetector::Query> adapter_sequences;
    for (const auto& adapter : adapters) {
        adapter_sequences.emplace_back(make_adapter_query(adapter));
    }
    auto result = m_adapter_sequences.emplace(kit_name, std::move(adapter_sequences));
    return result.first->second;
}

std::vector<AdapterDetector::Query>& AdapterDetector::get_primer_sequences(
        const std::string& kit_name) {
    auto it = m_primer_sequences.find(kit_name);
    if (it != m_primer_sequences.end()) {
        return it->second;
    }
    const auto& primers = get_primers_for_kit(kit_name);
    std::vector<AdapterDetector::Query> primer_sequences;
    for (const auto& primer : primers) {
        primer_sequences.emplace_back(make_primer_query(primer));
    }
    auto result = m_primer_sequences.emplace(kit_name, std::move(primer_sequences));
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
        std::string_view query_seq = queries[i].sequence;
        std::string_view query_seq_rev = queries[i].sequence_rev;
        spdlog::trace("Checking adapter/primer {}", name);

        if (!query_seq.empty()) {
            align(query_seq, read_front, name + "_FWD", front_results, IS_FRONT, placement_config);
        }
        if (!query_seq_rev.empty()) {
            align(query_seq_rev, read_rear, name + "_REV", rear_results, rear_start,
                  placement_config);
        }

        if (query_type == PRIMER) {
            // For primers we look for both the forward and reverse sequence at both ends.
            if (!query_seq_rev.empty()) {
                align(query_seq_rev, read_front, name + "_REV", front_results, IS_FRONT,
                      placement_config);
            }
            if (!query_seq.empty()) {
                align(query_seq, read_rear, name + "_FWD", rear_results, rear_start,
                      placement_config);
            }
        }
    }

    return {get_best_result(front_results), get_best_result(rear_results)};
}

}  // namespace demux
}  // namespace dorado
