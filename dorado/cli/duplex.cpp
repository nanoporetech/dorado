#include "Version.h"
#include "read_pipeline/DuplexCallerNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/duplex_utils.h"

#include <argparse.hpp>

#include <iostream>
#include <thread>
/*
#include "3rdparty/edlib/edlib/include/edlib.h"
*/

int duplex(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);
    parser.add_argument("reads_file").help("Basecalled reads.");
    parser.add_argument("pairs_file").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    std::cerr << "Loading BAM" << std::endl;

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::string reads_file = parser.get<std::string>("reads_file");
    std::string pairs_file = parser.get<std::string>("pairs_file");

    // Load the pairs file
    std::map<std::string, std::string> template_complement_map = load_pairs_file(pairs_file);

    // Load basecalls
    std::map<std::string, std::shared_ptr<Read>> reads = read_bam(reads_file);

    std::vector<std::string> args(argv, argv + argc);
    bool emit_fastq = parser.get<bool>("--emit-fastq");

    WriterNode writer_node(std::move(args), emit_fastq, 1);
    DuplexCallerNode duplex_caller_node(writer_node, template_complement_map, reads);

    /*    // Let's now perform alignment on all pairs:
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    for (auto key : template_complement_map) {
        std::string temp_id = key.first;
        std::string comp_id = key.second;

        std::vector<char> temp_str;
        std::vector<char> comp_str;
        std::vector<uint8_t> temp_q_string;
        if (reads.find(temp_id) == reads.end()) {
        } else {
            auto read = reads.at(temp_id);
            temp_str = read->sequence;
            temp_q_string = read->scores;
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

        if (reads.find(comp_id) == reads.end()) {
            //std::cerr << "Corresponding complement is missing" << std::endl;
        } else if (temp_str.size() != 0) {  // We can do the alignment
            auto complement_read = reads.at(comp_id);
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
                edlibFreeAlignResult(result);

                std::cout << ">" << temp_id << std::endl;
                for (auto& c : consensus) {
                    std::cout << c;
                }
                std::cout << std::endl;

                for (auto& q : q_string) {
                    std::cout << char(q + 33);
                }
                std::cout << std::endl;
            }
        }
    }*/
    return 0;
}