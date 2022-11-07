#include "DuplexCallerNode.h"

#include "3rdparty/edlib/edlib/include/edlib.h"

#include <spdlog/spdlog.h>

#include <chrono>

using namespace std::chrono_literals;

// Applies a min pool filter to q scores
void preprocess_quality_scores(std::vector<uint8_t>& quality_scores, int pool_window = 5) {
    // Apply a min-pool window to the quality scores
    auto opts = torch::TensorOptions().dtype(torch::kInt8);
    torch::Tensor t =
            torch::from_blob(quality_scores.data(), {1, (int)quality_scores.size()}, opts);
    auto t_float = t.to(torch::kFloat32);
    t.index({torch::indexing::Slice()}) =
            -torch::max_pool1d(-t_float, pool_window, 1, pool_window / 2);
}

// Returns reverse complement of a nucleotide sequence
void reverse_complement(std::vector<char>& sequence) {
    std::reverse(sequence.begin(), sequence.end());
    std::map<char, char> complementary_nucleotides = {
            {'A', 'T'}, {'C', 'G'}, {'G', 'C'}, {'T', 'A'}};
    std::for_each(sequence.begin(), sequence.end(),
                  [&complementary_nucleotides](char& c) { c = complementary_nucleotides[c]; });
}

// Given two sequences, their quality scores, and alignments, computes a consensus sequence
std::pair<std::vector<char>, std::vector<char>> compute_basespace_consensus(int alignment_start_position,
                                                                             int alignment_end_position,
                                                                             std::vector<uint8_t> target_quality_scores,
                                                                             int target_cursor,
                                                                             std::vector<uint8_t> query_quality_scores,
                                                                             int query_cursor,
                                                                             std::vector<char> target_sequence,
                                                                             std::vector<char> query_sequence,
                                                                             unsigned char* alignment) {
    std::vector<char> consensus;
    std::vector<char> quality_scores_phred;

    for (int i = alignment_start_position; i < alignment_end_position; i++) {
        if (target_quality_scores.at(target_cursor) >=
            query_quality_scores.at(query_cursor)) {  // Target has a higher quality score
            // If there is *not* an insertion to the query, add the nucleotide from the target cursor
            if (alignment[i] != 2) {
                consensus.push_back(target_sequence.at(target_cursor));
                quality_scores_phred.push_back(target_quality_scores.at(target_cursor) + 33);
            }
        } else {
            // If there is *not* an insertion to the target, add the nucleotide from the query cursor
            if (alignment[i] != 1) {
                consensus.push_back(
                        query_sequence.at(query_cursor));
                quality_scores_phred.push_back(query_quality_scores.at(query_cursor) + 33);
            }
        }

        //Anything but a query insertion and target advances
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

// Returns subset of alignment for which start and end start with  `num_consecutive_wanted` consecutive nucleotides.
std::pair<std::pair<int, int>, std::pair<int, int>> get_trimmed_alignment(int num_consecutive_wanted,
                                                                          unsigned char* alignment,
                                                                          int alignment_length,
                                                                          int target_cursor,
                                                                          int query_cursor,
                                                                          int start_alignment_position,
                                                                          int end_alignment_position){
    int num_consecutive = 0;

    // Find forward trim.
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[start_alignment_position] != 2) {
            target_cursor++;
        }

        if (alignment[start_alignment_position] != 1) {
            query_cursor++;
        }

        if (alignment[start_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;  //reset counter
        }

        start_alignment_position++;

        if (start_alignment_position >= alignment_length) {
            break;
        }

    }

    target_cursor -= num_consecutive_wanted;
    query_cursor -= num_consecutive_wanted;

    // Find reverse trim
    num_consecutive = 0;
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[end_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;
        }

        end_alignment_position--;

        if (end_alignment_position < start_alignment_position) {
            break;
        }
    }

    start_alignment_position -= num_consecutive_wanted;
    end_alignment_position += num_consecutive_wanted;

    auto alignment_start_end = std::make_pair(start_alignment_position, end_alignment_position);
    auto query_target_cursors = std::make_pair(query_cursor, target_cursor);

    return std::make_pair(alignment_start_end, query_target_cursors);

}

void DuplexCallerNode::worker_thread() {
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    // Loop over every template-complement pair, performing duplex calling on-the-fly
    for (auto key : m_template_complement_map) {
        std::string template_read_id = key.first;
        std::string complement_read_id = key.second;

        std::vector<char> template_sequence;
        std::shared_ptr<Read> template_read;
        std::vector<uint8_t> template_quality_scores;
        if (m_reads.find(template_read_id) == m_reads.end()) {
            spdlog::debug("Template Read ID=" + template_read_id +
                          "is present in pairs file but read was not found");
        } else {
            template_read = m_reads.at(template_read_id);
            template_sequence = template_read->sequence;
            template_quality_scores = template_read->scores;
        }

        preprocess_quality_scores(template_quality_scores);

        if (m_reads.find(complement_read_id) == m_reads.end()) {
            spdlog::debug("Complement ID=" + complement_read_id,
                          "paired with Template ID=" + template_read_id + " was not found");
        } else if (template_sequence.size() !=
                   0) {  // We have both sequences and can perform the consensus
            auto complement_read = m_reads.at(complement_read_id);
            std::vector<char> complement_str = complement_read->sequence;
            auto complement_quality_scores_reverse = complement_read->scores;
            std::reverse(complement_quality_scores_reverse.begin(),
                         complement_quality_scores_reverse.end());

            preprocess_quality_scores(complement_quality_scores_reverse);

            std::vector<char> complement_sequence_reverse_complement = complement_str;
            //compute the RC
            reverse_complement(complement_sequence_reverse_complement);

            EdlibAlignResult result =
                    edlibAlign(template_sequence.data(), template_sequence.size(),
                               complement_sequence_reverse_complement.data(),
                               complement_sequence_reverse_complement.size(), align_config);

            //Now - we have to do the actual basespace alignment itself
            int query_cursor = 0;
            int target_cursor = result.startLocations[0];

            auto [alignment_start_end, cursors] = get_trimmed_alignment(11,
                                                                        result.alignment,
                                                                        result.alignmentLength,
                                                                        target_cursor,
                                                                        query_cursor,
                                                                        0,
                                                                        result.endLocations[0]);

            query_cursor = cursors.first;
            target_cursor = cursors.second;
            int start_alignment_position = alignment_start_end.first;
            int end_alignment_position = alignment_start_end.second;

            if (start_alignment_position < end_alignment_position) {
                auto [consensus, quality_scores_phred] = compute_basespace_consensus(start_alignment_position,
                                                                                     end_alignment_position,
                                                                                     template_quality_scores,
                                                                                     target_cursor,
                                                                                     complement_quality_scores_reverse,
                                                                                     query_cursor,
                                                                                     template_sequence,
                                                                                     complement_sequence_reverse_complement,
                                                                                     result.alignment);
                auto duplex_read = std::make_shared<Read>();
                duplex_read->seq = std::string( consensus.begin(), consensus.end());
                duplex_read->qstring = std::string(quality_scores_phred.begin(), quality_scores_phred.end());
                duplex_read->read_id = template_read->read_id + ";" + complement_read->read_id;

                m_sink.push_read(duplex_read);
            }
            edlibFreeAlignResult(result);
        }
    }
}

DuplexCallerNode::DuplexCallerNode(ReadSink& sink,
                                   std::map<std::string, std::string> template_complement_map,
                                   std::map<std::string, std::shared_ptr<Read>> reads)
        : ReadSink(1000),
          m_sink(sink),
          m_template_complement_map(template_complement_map),
          m_reads(reads) {
    // Only one worker at present
    for (int i = 0; i < 1; i++) {
        std::unique_ptr<std::thread> worker_thread =
                std::make_unique<std::thread>(&DuplexCallerNode::worker_thread, this);
        worker_threads.push_back(std::move(worker_thread));
    }
}

DuplexCallerNode::~DuplexCallerNode() {
    terminate();
    m_cv.notify_one();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : worker_threads) {
        t->join();
    }

    //Notify the sink that the Node has terminated
    m_sink.terminate();
}