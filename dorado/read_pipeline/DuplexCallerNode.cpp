#include "DuplexCallerNode.h"

#include "3rdparty/edlib/edlib/include/edlib.h"

#include <spdlog/spdlog.h>

#include <chrono>

using namespace std::chrono_literals;

void preprocess_quality_scores(std::vector<uint8_t>& quality_scores, int pool_window = 5) {
    // Apply a min-pool window to the quality scores
    auto opts = torch::TensorOptions().dtype(torch::kInt8);
    torch::Tensor t =
            torch::from_blob(quality_scores.data(), {1, (int)quality_scores.size()}, opts);
    auto t_float = t.to(torch::kFloat32);
    t.index({torch::indexing::Slice()}) =
            -torch::max_pool1d(-t_float, pool_window, 1, pool_window / 2);
}

void reverse_complement(std::vector<char>& sequence) {
    std::reverse(sequence.begin(), sequence.end());
    std::map<char, char> complementary_nucleotides = {
            {'A', 'T'}, {'C', 'G'}, {'G', 'C'}, {'T', 'A'}};
    std::for_each(sequence.begin(), sequence.end(),
                  [&complementary_nucleotides](char& c) { c = complementary_nucleotides[c]; });
}

void compute_basespace_consensus() {}

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
            spdlog::error("Template Read ID=", template_read_id,
                          "is present in pairs file but read was not found");
        } else {
            template_read = m_reads.at(template_read_id);
            template_sequence = template_read->sequence;
            template_quality_scores = template_read->scores;
        }

        preprocess_quality_scores(template_quality_scores);

        if (m_reads.find(complement_read_id) == m_reads.end()) {
            spdlog::debug("Complement ID=", complement_read_id,
                          "paired with Template ID=", template_read_id, " is missing");
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

            //Let's do the start trim
            int num_consecutive = 0;
            int num_consecutive_wanted = 11;
            int alignment_position = 0;
            bool alignment_possible = true;

            // Find forward trim.
            while (num_consecutive < num_consecutive_wanted) {
                if (result.alignment[alignment_position] != 2) {
                    target_cursor++;
                }

                if (result.alignment[alignment_position] != 1) {
                    query_cursor++;
                }

                if (result.alignment[alignment_position] == 0) {
                    num_consecutive++;
                } else {
                    num_consecutive = 0;  //reset counter
                }

                alignment_position++;

                if (alignment_position >= result.alignmentLength) {
                    alignment_possible = false;
                    break;
                }
            }

            target_cursor -= num_consecutive_wanted;
            query_cursor -= num_consecutive_wanted;

            // Find reverse trim
            int end_alignment_position = result.endLocations[0];
            num_consecutive = 0;
            while (num_consecutive < num_consecutive_wanted) {
                if (result.alignment[end_alignment_position] == 0) {
                    num_consecutive++;
                } else {
                    num_consecutive = 0;
                }

                end_alignment_position--;

                if (end_alignment_position < alignment_position) {
                    alignment_possible = false;
                    break;
                }
            }

            alignment_position -= num_consecutive_wanted;
            end_alignment_position += num_consecutive_wanted;

            std::vector<char> consensus;
            std::vector<char> q_string;

            if (alignment_possible) {
                for (int i = alignment_position; i < end_alignment_position; i++) {
                    if (template_quality_scores.at(target_cursor) >=
                        complement_quality_scores_reverse.at(query_cursor)) {  // Target is higher Q
                        // If there is *not* an insertion to the query, add the nucleotide from the target cursor
                        if (result.alignment[i] != 2) {
                            consensus.push_back(template_sequence.at(target_cursor));
                            q_string.push_back(template_quality_scores.at(target_cursor));
                        }
                    } else {
                        // If there is *not* an insertion to the target, add the nucleotide from the query cursor
                        if (result.alignment[i] != 1) {
                            consensus.push_back(
                                    complement_sequence_reverse_complement.at(query_cursor));
                            q_string.push_back(complement_quality_scores_reverse.at(query_cursor));
                        }
                    }

                    //Anything but a query insertion and target advances
                    if (result.alignment[i] != 2) {
                        target_cursor++;
                    }

                    //Anything but a target insertion and query advances
                    if (result.alignment[i] != 1) {
                        query_cursor++;
                    }
                }

                auto duplex_read = std::make_shared<Read>();
                duplex_read->seq = std::string(consensus.begin(), consensus.end());
                for (auto& q : q_string) {
                    q += 33;
                }
                duplex_read->qstring = std::string(q_string.begin(), q_string.end());

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
    for (int i = 0; i < 1; i++) {  //TODO fix this its silly
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