#include "3rdparty/edlib/edlib/include/edlib.h"

#include "DuplexCallerNode.h"
#include <chrono>

using namespace std::chrono_literals;

void DuplexCallerNode::worker_thread() {

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    for (auto key : m_template_complement_map) {
        std::string temp_id = key.first;
        std::string comp_id = key.second;

        std::vector<char> temp_str;
        std::vector<char> comp_str;
        std::vector<uint8_t> temp_q_string;
        std::shared_ptr<Read> template_read;
        if (m_reads.find(temp_id) == m_reads.end()) {
        } else {
            template_read = m_reads.at(temp_id);
            temp_str = template_read->sequence;
            temp_q_string = template_read->scores;
        }

        auto opts = torch::TensorOptions().dtype(torch::kInt8);
        torch::Tensor t = torch::from_blob(temp_q_string.data(),
                                           {1, (int) temp_q_string.size()}, opts);
        auto t_float = t.to(torch::kFloat32);
        int pool_window = 5;
        t.index({torch::indexing::Slice()}) = -torch::max_pool1d(-t_float,
                                                                 pool_window,
                                                                 1,
                                                                 pool_window / 2);

        if (m_reads.find(comp_id) == m_reads.end()) {
            //std::cerr << "Corresponding complement is missing" << std::endl;
        } else if (temp_str.size() != 0) {  // We can do the alignment
            auto complement_read = m_reads.at(comp_id);
            comp_str = complement_read->sequence;
            auto comp_q_scores_reverse = complement_read->scores;

            auto opts = torch::TensorOptions().dtype(torch::kInt8);
            torch::Tensor t = torch::from_blob(comp_q_scores_reverse.data(),
                                               {1, (int)comp_q_scores_reverse.size()}, opts);
            auto t_float = t.to(torch::kFloat32);
            int pool_window = 5;
            t.index({torch::indexing::Slice()}) =
                    -torch::max_pool1d(-t_float, pool_window, 1, pool_window / 2);

            std::reverse(comp_q_scores_reverse.begin(), comp_q_scores_reverse.end());

            std::vector<char> comp_str_rc = comp_str;
            //compute the RC
            std::reverse(comp_str_rc.begin(), comp_str_rc.end());

            std::map<char,char> complementary_nucleotides = { {'A', 'T'},
                                                              {'C', 'G'},
                                                              {'G', 'C'},
                                                              {'T', 'A'}};
            std::for_each(
                    comp_str_rc.begin(), comp_str_rc.end(),
                    [&complementary_nucleotides](char& c) { c = complementary_nucleotides[c]; });

            EdlibAlignResult result =
                    edlibAlign(temp_str.data(), temp_str.size(), comp_str_rc.data(),
                               comp_str_rc.size(), align_config);

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
            while(num_consecutive < num_consecutive_wanted){
                if (result.alignment[end_alignment_position] == 0){
                    num_consecutive++;
                } else {
                    num_consecutive = 0;
                }

                end_alignment_position--;

                if (end_alignment_position < alignment_position){
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
                    if (temp_q_string.at(target_cursor) >= comp_q_scores_reverse.at(query_cursor)) { // Target is higher Q
                        // If there is *not* an insertion to the query, add the nucleotide from the target cursor
                        if (result.alignment[i] != 2) {
                            consensus.push_back(temp_str.at(target_cursor));
                            q_string.push_back(temp_q_string.at(target_cursor));
                        }
                    } else {
                        // If there is *not* an insertion to the target, add the nucleotide from the query cursor
                        if (result.alignment[i] != 1) {
                            consensus.push_back(comp_str_rc.at(query_cursor));
                            q_string.push_back(comp_q_scores_reverse.at(query_cursor));
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

                template_read->seq = std::string(consensus.begin(), consensus.end());

                for (auto& q : q_string) {
                    q += 33;
                }

                template_read->qstring = std::string(q_string.begin(), q_string.end());

                m_sink.push_read(template_read);
            }
            edlibFreeAlignResult(result);
        }
    }
}

DuplexCallerNode::DuplexCallerNode(ReadSink& sink,
                                   std::map<std::string, std::string> template_complement_map,
                                   std::map<std::string, std::shared_ptr<Read>> reads)
        : ReadSink(1000), m_sink(sink),
          m_template_complement_map(template_complement_map),
          m_reads(reads)
{
    for (int i = 0; i < 1; i++) { //TODO fix this its silly
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