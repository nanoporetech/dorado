#include "AdapterDetector.h"

#include "utils/alignment_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <edlib.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>


namespace {

const int TRIM_LENGTH = 150;

// Create edlib configuration for detecting adapters and primers.
EdlibAlignConfig init_edlib_config_for_adapters() {
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_PATH;
    // The Ns are the barcode mask.
    static const EdlibEqualityPair additionalEqualities[4] = {
            {'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 4;
    return placement_config;
}

// For adapters, we there are specific sequences we look for at the front of the read. We don't look for exactly
// the reverse complement at the rear of the read, though, because it will generally be truncated. So we list here
// the specific sequences to look for at the front and rear of the reads.
struct Adapter {
    std::string name;
    std::string front_sequence;
    std::string rear_sequence;
};

const std::vector<Adapter> adapters = {
    {"LSK109", "AATGTACTTCGTTCAGTTACGTATTGCT", "AGCAATACGTAACTGAACGAAGT"},
    {"LSK110", "CCTGTACTTCGTTCAGTTACGTATTGC", "AGCAATACGTAACTGAAC"}
};

// For primers, we look for each primer sequence, and its reverse complement, at both the front and rear of the read.
struct Primer {
    std::string name;
    std::string sequence;
};

const std::vector<Primer> primers = {
    {"PCR_PSK_rev1", "ACTTGCCTGTCGCTCTATCTTCGGCGTCTGCTTGGGTGTTTAACC"},
    {"PCR_PSK_rev2", "TTTCTGTTGGTGCTGATATTGCGGCGTCTGCTTGGGTGTTTAACCT"},
    {"cDNA_VNP", "ACTTGCCTGTCGCTCTATCTTC"},
    {"cDNA_SSP", "TTTCTGTTGGTGCTGATATTGCTGGG"},
    {"PCS110_forward", "TCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTT"},
    {"PCS110_reverse", "ATCGCCTACCGTGACAAGAAAGTTGTCGGTGTCTTTGTGTTTCTGTTGGTGCTGATATTGCTTT"},
    {"RAD", "GCTTGGGTGTTTAACCGTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA"}
};

}  // namespace

namespace dorado {
namespace demux {

AdapterDetector::AdapterDetector() {
    m_adapter_sequences.resize(adapters.size());
    for (size_t i = 0; i < adapters.size(); ++i) {
        m_adapter_sequences[i].name = adapters[i].name;
        m_adapter_sequences[i].sequence = adapters[i].front_sequence;
        m_adapter_sequences[i].sequence_rev = adapters[i].rear_sequence;
    }
    // We need 2 entries for each primer, because
    m_primer_sequences.resize(primers.size());
    for (size_t i = 0; i < primers.size(); ++i) {
        m_primer_sequences[i].name = primers[i].name;
        m_primer_sequences[i].sequence = primers[i].sequence;
        m_primer_sequences[i].sequence_rev = utils::reverse_complement(primers[i].sequence);
        ++i;
    }
}

AdapterDetector::~AdapterDetector() = default;

AdapterScoreResult AdapterDetector::find_adapters(const std::string& seq) {
    return detect(seq, m_adapter_sequences, ADAPTER);
}

AdapterScoreResult AdapterDetector::find_primers(const std::string& seq) {
    return detect(seq, m_primer_sequences, PRIMER);
}

static SingleEndResult copy_results(const EdlibAlignResult& source, const std::string& name, size_t length) {
    SingleEndResult dest;
    dest.name = name;
    dest.score = 1.0f - float(source.editDistance) / length;
    dest.position = {source.startLocations[0], source.endLocations[0]};
    return dest;
}

AdapterScoreResult AdapterDetector::detect(
        const std::string& seq,
        const std::vector<Query>& queries,
        AdapterDetector::QueryType query_type) const {
    std::string_view read_front = seq.substr(0, TRIM_LENGTH);
    int rear_start = std::max(0, int(seq.length()) - TRIM_LENGTH);
    std::string_view read_rear = seq.substr(rear_start, TRIM_LENGTH);

    // Try to find the location of the queries in the front and rear windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_adapters();

    std::vector<SingleEndResult> front_results, rear_results;
    for (size_t i = 0; i < queries.size(); i++) {
        const auto& name = queries[i].name;
        const auto& sequence = queries[i].sequence;
        const auto& sequence_rev = queries[i].sequence_rev;
        spdlog::debug("Checking adapter/primer {}", name);

        auto front_result = edlibAlign(sequence.data(), int(sequence.length()), read_front.data(), int(read_front.length()), placement_config);
        front_results.emplace_back(copy_results(front_result, name + "_FWD", sequence.length()));
        edlibFreeAlignResult(front_result);
        if (query_type == PRIMER) {
            // For primers we look for both the forward and reverse sequence at both ends.
            auto front_result_rev = edlibAlign(sequence_rev.data(), int(sequence_rev.length()), read_front.data(), int(read_front.length()), placement_config);
            front_results.emplace_back(copy_results(front_result_rev, name + "_REV", sequence_rev.length()));
            edlibFreeAlignResult(front_result_rev);
        }
        
        auto rear_result = edlibAlign(sequence_rev.data(), int(sequence_rev.length()), read_rear.data(), int(read_rear.length()), placement_config);
        rear_results.emplace_back(copy_results(rear_result, name + "_REV", sequence_rev.length()));
        edlibFreeAlignResult(rear_result);
        if (query_type == PRIMER) {
            auto rear_result_fwd = edlibAlign(sequence.data(), int(sequence.length()), read_rear.data(), int(read_rear.length()), placement_config);
            rear_results.emplace_back(copy_results(rear_result_fwd, name + "_FWD", sequence.length()));
            edlibFreeAlignResult(rear_result_fwd);
        }
    }
    int best_front = -1, best_rear = -1;
    float best_front_score = -1.0f, best_rear_score = -1.0f;
    AdapterScoreResult result;
    for (size_t i = 0; i < front_results.size(); ++i) {
        if (front_results[i].score > best_front_score) {
            best_front_score = front_results[i].score;
            best_front = int(i); 
        }
    }
    for (size_t i = 0; i < rear_results.size(); ++i) {
        if (rear_results[i].score > best_rear_score) {
            best_rear_score = rear_results[i].score;
            best_rear = int(i);
        }
    }
    result.front = front_results[best_front];
    result.rear = rear_results[best_rear];
    return result;
}

}  // namespace demux
}  // namespace dorado
