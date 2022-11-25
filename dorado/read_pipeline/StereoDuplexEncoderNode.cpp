#include "StereoDuplexEncoderNode.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/duplex_utils.h"

#include <chrono>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace {
std::shared_ptr<dorado::Read> stereo_encode(std::shared_ptr<dorado::Read> template_read,
                                            std::shared_ptr<dorado::Read> complement_read) {
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    std::vector<char> complement_sequence_reverse_complement(complement_read->seq.begin(),
                                                             complement_read->seq.end());
    dorado::utils::reverse_complement(complement_sequence_reverse_complement);

    std::vector<uint8_t> complemnet_q_scores_reversed(complement_read->qstring.begin(),
                                                      complement_read->qstring.end());
    std::reverse(complemnet_q_scores_reversed.begin(), complemnet_q_scores_reversed.end());

    std::vector<char> template_sequence(template_read->seq.begin(), template_read->seq.end());
    std::vector<uint8_t> template_q_scores(template_read->qstring.begin(),
                                           template_read->qstring.end());

    // Align the two reads to one another and print out the score.
    EdlibAlignResult result =
            edlibAlign(template_read->seq.data(), template_read->seq.size(),
                       complement_sequence_reverse_complement.data(),
                       complement_sequence_reverse_complement.size(), align_config);

    if (template_read->read_id == "025c2ee5-6775-43b8-85cb-0397fef88d5b") {
        std::cerr << "Edit distance is " << result.editDistance << std::endl;
        std::cerr << "Template sequence is " << template_read->seq << std::endl;
        std::cerr << "Complement sequence is " << complement_read->seq << std::endl;
    }

    int query_cursor = 0;
    int target_cursor = result.startLocations[0];

    auto [alignment_start_end, cursors] = dorado::utils::get_trimmed_alignment(
            11, result.alignment, result.alignmentLength, target_cursor, query_cursor, 0,
            result.endLocations[0]);

    query_cursor = cursors.first;
    target_cursor = cursors.second;
    int start_alignment_position = alignment_start_end.first;
    int end_alignment_position = alignment_start_end.second;

    if (template_read->read_id == "025c2ee5-6775-43b8-85cb-0397fef88d5b") {
        std::cerr << "start_alignment_position " << start_alignment_position << std::endl;
        std::cerr << "end_alignment_position " << end_alignment_position << std::endl;
        std::cerr << "Target cursor " << target_cursor << std::endl;
        std::cerr << "Query cursor " << query_cursor << std::endl;
    }

    // TODO: perhaps its overkill having this function make this decision...
    int min_trimmed_alignment_length = 200;
    bool consensus_possible =
            (start_alignment_position < end_alignment_position) &&
            ((end_alignment_position - start_alignment_position) > min_trimmed_alignment_length);

    int stride = 5;  // TODO this needs to be passed in as a parameter
    std::shared_ptr<dorado::Read> read = std::make_shared<dorado::Read>();
    if (consensus_possible) {
        //dorado::utils::preprocess_quality_scores(template_q_scores);
        //dorado::utils::preprocess_quality_scores(complemnet_q_scores_reversed);

        // Step 3 - Move along the alignment, filling out the stereo-encoded tensor
        // Prepare the encoding tensor
        int max_size = template_read->raw_data.size(0) + template_read->raw_data.size(0);
        auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
        int num_features = 13;
        auto tmp = torch::zeros({num_features, max_size}, opts);

        // Diagnostics one: Is sum of move vector the same length as the sequence
        int num_moves = 0;
        for (int i = 0; i < template_read->moves.size(); i++) {
            num_moves += template_read->moves[i];
        }
        std::cerr << "number of moves is: " << num_moves << std::endl;
        std::cerr << "sequence length is: " << template_read->seq.size() << std::endl;

        // Step one - we need three cursors (besides the sequence ones)
        // 1. TSignal cursor
        // 2. CSignal Cursor
        // 3. Encoding cursor

        // Step 1 - let's get the tsignal cursor.
        // We have the target_query, let's loop over the move table until we have got our index into the signal.

        int template_signal_cursor = 0;
        int complement_signal_cursor = 0;

        std::vector<uint8_t> template_moves_expanded;
        for (int i = 0; i < template_read->moves.size(); i++) {
            template_moves_expanded.push_back(template_read->moves[i]);
            for (int j = 0; j < stride - 1; j++) {
                template_moves_expanded.push_back(0);
            }
        }  // TODO add the last elements
        int extra_moves = template_moves_expanded.size() - template_read->raw_data.size(0);
        for (int i = 0; i < extra_moves; i++) {
            template_moves_expanded.pop_back();
        }

        std::cerr << "Raw data size: " << template_read->raw_data.size(0) << std::endl;
        std::cerr << "Expanded move table size: " << template_moves_expanded.size() << std::endl;

        int template_moves_seen = template_moves_expanded[template_signal_cursor];
        while (template_moves_seen < target_cursor + 1) {  // TODO: is this +1 correct?
            template_signal_cursor++;
            template_moves_seen += template_moves_expanded[template_signal_cursor];
        }

        std::vector<uint8_t> complement_moves_expanded;
        for (int i = 0; i < complement_read->moves.size(); i++) {
            complement_moves_expanded.push_back(complement_read->moves[i]);
            for (int j = 0; j < stride - 1; j++) {
                complement_moves_expanded.push_back(0);
            }
        }  // TODO add the last elements

        extra_moves = complement_moves_expanded.size() - complement_read->raw_data.size(0);
        for (int i = 0; i < extra_moves; i++) {
            complement_moves_expanded.pop_back();
        }
        std::cerr << "Raw data size: " << complement_read->raw_data.size(0) << std::endl;
        std::cerr << "Expanded move table size: " << complement_moves_expanded.size() << std::endl;

        complement_moves_expanded.push_back(1);
        std::reverse(complement_moves_expanded.begin(), complement_moves_expanded.end());
        complement_moves_expanded.pop_back();

        auto complement_signal = complement_read->raw_data;
        complement_signal = torch::flip(complement_read->raw_data, 0);

        int complement_moves_seen = complement_read->moves[complement_signal_cursor];
        while (complement_moves_seen < query_cursor + 1) {
            complement_signal_cursor++;
            complement_moves_seen += complement_moves_expanded[complement_signal_cursor];
        }

        std::cerr << complement_moves_expanded << std::endl;

        float pad_value = 0.8 * std::min(torch::min(complement_signal).item<float>(),
                                         torch::min(template_read->raw_data).item<float>());

        tmp.index({torch::indexing::Slice(None, 2)}) = pad_value;

        int stereo_global_cursor = 0;  // Index into the stereo-encoded signal
        for (int i = start_alignment_position; i < end_alignment_position; i++) {
            // We move along every alignment position. For every position we need to add signal and padding.
            //Let us add the respective nucleotides and q-scores

            int template_segment_length = 0;    // index into this segment in signal-space
            int complement_segment_length = 0;  // index into this segment in signal-space

            // If there is *not* an insertion to the query, add the nucleotide from the target cursor
            if (result.alignment[i] != 2) {
                //need to initialise
                tmp[0][template_segment_length + stereo_global_cursor] =
                        template_read->raw_data[template_signal_cursor];
                template_segment_length++;
                template_signal_cursor++;
                auto max_signal_length = template_moves_expanded.size();
                while (template_moves_expanded[template_signal_cursor] == 0 &&
                       (template_signal_cursor < max_signal_length)) {
                    tmp[0][stereo_global_cursor + template_segment_length] =
                            template_read->raw_data[template_signal_cursor];
                    template_signal_cursor++;
                    template_segment_length++;
                }
            }

            // If there is *not* an insertion to the target, add the nucleotide from the query cursor
            if (result.alignment[i] != 1) {
                char nucleotide = complement_sequence_reverse_complement.at(query_cursor);
                //need to initialise
                tmp[1][complement_segment_length + stereo_global_cursor] =
                        complement_signal[complement_signal_cursor];

                complement_segment_length++;
                complement_signal_cursor++;
                auto max_signal_length = complement_moves_expanded.size();
                while (complement_moves_expanded[complement_signal_cursor] == 0 &&
                       (complement_signal_cursor < max_signal_length)) {
                    tmp[1][stereo_global_cursor + complement_segment_length] =
                            complement_signal[complement_signal_cursor];
                    complement_signal_cursor++;
                    complement_segment_length++;
                }
            }

            int total_segment_length = std::max(template_segment_length, complement_segment_length);

            // Now, add the nucleotides and q scores
            if (result.alignment[i] != 2) {
                char nucleotide = template_sequence.at(target_cursor);
                for (int i = 0; i < total_segment_length; i++) {
                    tmp[2 + (0b11 & (nucleotide >> 2 ^ nucleotide >> 1))]
                       [stereo_global_cursor + i] = 1;
                    tmp[11][stereo_global_cursor + i] =
                            float(template_q_scores.at(target_cursor) - 33) / 90;
                }
            }

            // Now, add the nucleotides and q scores
            if (result.alignment[i] != 1) {
                char nucleotide = complement_sequence_reverse_complement.at(query_cursor);
                for (int i = 0; i < total_segment_length; i++) {
                    tmp[6 + (0b11 & (nucleotide >> 2 ^ nucleotide >> 1))]
                       [stereo_global_cursor + i] = 1;
                    tmp[12][stereo_global_cursor + i] =
                            float(complemnet_q_scores_reversed.at(query_cursor) - 33) / 90;
                }
            }

            tmp[10][stereo_global_cursor] = 1;  //set the move table

            //update the global cursor
            stereo_global_cursor += total_segment_length;
            // Update query and target cursors
            //Anything excluding a query insertion causes the target cursor to advance
            if (result.alignment[i] != 2) {
                target_cursor++;
            }

            //Anything but a target insertion and query advances
            if (result.alignment[i] != 1) {
                query_cursor++;
            }
        }

        tmp = tmp.index({torch::indexing::Slice(None), torch::indexing::Slice(None, 200)});

        std::cerr << tmp << std::endl;

        return read;
    }
    return read;
}
}  // namespace

namespace dorado {
// Let's make a stub which just consumes reads from its input queue and passes it to its output queue.
// Next step

void StereoDuplexEncoderNode::worker_thread() {
    int i = 0;

    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Termination flag is set and read input queue is empty, so terminate the worker
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        bool read_is_template = false;
        bool partner_found = false;
        std::string partner_id;

        // Check if read is a template with corresponding complement
        if (m_template_complement_map.find(read->read_id) != m_template_complement_map.end()) {
            read_is_template = true;
            partner_id = m_template_complement_map[read->read_id];
            partner_found = true;
        } else {
            if (m_complement_template_map.find(read->read_id) != m_complement_template_map.end()) {
                partner_id = m_complement_template_map[read->read_id];
                partner_found = true;
            }
        }

        if (partner_found) {
            i++;
            if (i % 100 == 0) {
                std::cerr << i << std::endl;
            }

            if (read_cache.find(partner_id) == read_cache.end()) {
                // Partner is not in the read cache
                read_cache[read->read_id] = read;
            } else {
                std::shared_ptr<Read> template_read;
                std::shared_ptr<Read> complement_read;

                auto partner_read_itr = read_cache.find(partner_id);
                auto partner_read = partner_read_itr->second;
                read_cache.erase(partner_read_itr);

                if (read_is_template) {
                    template_read = read;
                    complement_read = partner_read;
                } else {
                    complement_read = read;
                    template_read = partner_read;
                }

                std::shared_ptr<Read> stereo_encoded_read =
                        stereo_encode(template_read, complement_read);
                if (stereo_encoded_read->seq.size() > 100) {
                    //m_sink.push_read(stereo_encoded_read);  // Found a partner, so process it.
                }
            }
        }
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(
        ReadSink &sink,
        std::map<std::string, std::string> template_complement_map)
        : ReadSink(1000), m_sink(sink), m_template_complement_map(template_complement_map) {
    // Set up teh complement_template_map

    for (auto key : template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    std::unique_ptr<std::thread> stereo_encoder_worker_thread =
            std::make_unique<std::thread>(&StereoDuplexEncoderNode::worker_thread, this);
    worker_threads.push_back(std::move(stereo_encoder_worker_thread));
}

StereoDuplexEncoderNode::~StereoDuplexEncoderNode() {
    terminate();
    m_cv.notify_one();
    for (auto &t : worker_threads) {
        t->join();
    }
}

}  // namespace dorado
