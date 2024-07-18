#include "BaseSpaceDuplexCallerNode.h"

#include "torch_utils/duplex_utils.h"
#include "utils/sequence_utils.h"
#include "utils/thread_naming.h"

#include <cxxpool.h>
#include <edlib.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

using namespace std::chrono_literals;
namespace {
// Given two sequences, their quality scores, and alignments, computes a consensus sequence
std::pair<std::vector<char>, std::vector<char>> compute_basespace_consensus(
        int alignment_start_position,
        int alignment_end_position,
        std::vector<uint8_t> target_quality_scores,
        int target_cursor,
        std::vector<uint8_t> query_quality_scores,
        int query_cursor,
        const std::string_view target_sequence,
        const std::string_view query_sequence,
        unsigned char* alignment) {
    std::vector<char> consensus;
    std::vector<char> quality_scores_phred;

    // Loop over each alignment position, within given alignment boundaries
    for (int i = alignment_start_position;
         i < alignment_end_position && target_cursor < int(target_quality_scores.size()) &&
         query_cursor < int(query_quality_scores.size());
         i++) {
        //Comparison between q-scores is done in Phred space which is offset by 33
        if (target_quality_scores.at(target_cursor) >=
            query_quality_scores.at(query_cursor)) {  // Target has a higher quality score
            // If there is *not* an insertion to the query, add the nucleotide from the target cursor
            if (alignment[i] != 2) {
                consensus.push_back(target_sequence.at(target_cursor));
                quality_scores_phred.push_back(target_quality_scores.at(target_cursor));
            }
        } else {
            // If there is *not* an insertion to the target, add the nucleotide from the query cursor
            if (alignment[i] != 1) {
                consensus.push_back(query_sequence.at(query_cursor));
                quality_scores_phred.push_back(query_quality_scores.at(query_cursor));
            }
        }

        //Anything excluding a query insertion causes the target cursor to advance
        if (alignment[i] != 2) {
            target_cursor++;
        }

        //Anything but a target insertion and query advances
        if (alignment[i] != 1) {
            query_cursor++;
        }
    }
    return std::make_pair(consensus, quality_scores_phred);
}
}  // namespace

namespace dorado {

void BaseSpaceDuplexCallerNode::worker_thread() {
    utils::set_thread_name("duplex_worker");
    cxxpool::thread_pool pool{m_num_worker_threads};
    std::vector<std::future<void>> futures;

    futures.reserve(m_template_complement_map.size());
    for (const auto& key : m_template_complement_map) {
        futures.push_back(pool.push([key, this] { return basespace(key.first, key.second); }));
    }
    for (auto& v : futures) {
        v.get();
    }
}

void BaseSpaceDuplexCallerNode::basespace(const std::string& template_read_id,
                                          const std::string& complement_read_id) {
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    std::string_view template_sequence;
    const SimplexRead* template_read;
    std::vector<uint8_t> template_quality_scores;
    auto template_read_it = m_reads.find(template_read_id);
    if (template_read_it == m_reads.end()) {
        spdlog::debug("Template Read ID={} is present in pairs file but read was not found",
                      template_read_id);
        return;
    } else {
        template_read = template_read_it->second.get();
        template_sequence = template_read->read_common.seq;
        template_quality_scores = std::vector<uint8_t>(template_read->read_common.qstring.begin(),
                                                       template_read->read_common.qstring.end());
    }

    // For basespace, a q score filter is run over the quality scores.
    utils::preprocess_quality_scores(template_quality_scores);

    auto complement_read_it = m_reads.find(complement_read_id);
    if (complement_read_it == m_reads.end()) {
        spdlog::debug("Complement ID={} paired with Template ID={} was not found",
                      complement_read_id, template_read_id);
        return;
    }

    if (template_sequence.empty()) {
        return;
    }

    // We have both sequences and can perform the consensus
    const SimplexRead* complement_read = complement_read_it->second.get();
    auto complement_quality_scores_reverse =
            std::vector<uint8_t>(complement_read->read_common.qstring.begin(),
                                 complement_read->read_common.qstring.end());
    std::reverse(complement_quality_scores_reverse.begin(),
                 complement_quality_scores_reverse.end());

    // For basespace, a q score filter is run over the quality scores.
    utils::preprocess_quality_scores(complement_quality_scores_reverse);

    // Compute the RC
    auto complement_sequence_reverse_complement =
            dorado::utils::reverse_complement(complement_read->read_common.seq);

    EdlibAlignResult result =
            edlibAlign(template_sequence.data(), int(template_sequence.size()),
                       complement_sequence_reverse_complement.data(),
                       int(complement_sequence_reverse_complement.size()), align_config);

    // Now - we have to do the actual basespace alignment itself
    int query_cursor = 0;
    int target_cursor =
            result.startLocations[0];  // 0-based position in the *target* where alignment starts.

    // Adjust min consecutive wanted based on sequence lengths. If reads are short (< 500bp), use an overlap of 5, otherwise use 11.
    const int kMinNumConsecutiveWanted =
            (std::min(template_sequence.size(), complement_sequence_reverse_complement.size()) < 500
                     ? 5
                     : 11);
    auto [alignment_start_end, cursors] = utils::get_trimmed_alignment(
            kMinNumConsecutiveWanted, result.alignment, result.alignmentLength, target_cursor,
            query_cursor, 0, result.endLocations[0]);

    query_cursor = cursors.first;
    target_cursor = cursors.second;
    int start_alignment_position = alignment_start_end.first;
    int end_alignment_position = alignment_start_end.second;

    const int kMinTrimmedAlignmentLength =
            (std::min(template_sequence.size(), complement_sequence_reverse_complement.size()) < 500
                     ? 25
                     : 200);
    bool consensus_possible =
            (start_alignment_position < end_alignment_position) &&
            ((end_alignment_position - start_alignment_position) > kMinTrimmedAlignmentLength);

    if (consensus_possible) {
        auto [consensus, quality_scores_phred] = compute_basespace_consensus(
                start_alignment_position, end_alignment_position, template_quality_scores,
                target_cursor, complement_quality_scores_reverse, query_cursor, template_sequence,
                complement_sequence_reverse_complement, result.alignment);

        auto duplex_read = std::make_unique<DuplexRead>();
        duplex_read->read_common.is_duplex = true;
        duplex_read->read_common.seq = std::string(consensus.begin(), consensus.end());
        duplex_read->read_common.qstring =
                std::string(quality_scores_phred.begin(), quality_scores_phred.end());

        duplex_read->read_common.read_id =
                template_read->read_common.read_id + ";" + complement_read->read_common.read_id;
        duplex_read->read_common.read_tag = template_read->read_common.read_tag;

        send_message_to_sink(std::move(duplex_read));
    }
    edlibFreeAlignResult(result);
}

BaseSpaceDuplexCallerNode::BaseSpaceDuplexCallerNode(
        std::map<std::string, std::string> template_complement_map,
        ReadMap reads,
        size_t threads)
        : MessageSink(1000, 0),
          m_num_worker_threads(threads),
          m_template_complement_map(std::move(template_complement_map)),
          m_reads(std::move(reads)) {}

void BaseSpaceDuplexCallerNode::start_threads() {
    m_worker_thread = std::thread([this] { worker_thread(); });
}

void BaseSpaceDuplexCallerNode::terminate_impl() {
    terminate_input_queue();
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
}

void BaseSpaceDuplexCallerNode::restart() {
    start_input_queue();
    start_threads();
}

}  // namespace dorado
