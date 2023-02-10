#include "StereoDuplexEncoderNode.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/duplex_utils.h"

#include <chrono>
#include <cstring>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace stereo_internal {
std::shared_ptr<dorado::Read> stereo_encode(std::shared_ptr<dorado::Read> template_read,
                                            std::shared_ptr<dorado::Read> complement_read) {
    // We rely on the incoming read raw data being of type float32 to allow dumb copying.
    assert(template_read->raw_data.dtype() == torch::kFloat32);
    assert(complement_read->raw_data.dtype() == torch::kFloat32);

    std::shared_ptr<dorado::Read> read = std::make_shared<dorado::Read>();  // Return read

    float template_len = template_read->seq.size();
    float complement_len = complement_read->seq.size();

    float delta = std::max(template_len, complement_len) - std::min(template_len, complement_len);
    if ((delta / std::max(template_len, complement_len)) > 0.05) {
        return read;
    }

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    std::vector<char> complement_sequence_reverse_complement(complement_read->seq.begin(),
                                                             complement_read->seq.end());
    dorado::utils::reverse_complement(complement_sequence_reverse_complement);

    std::vector<uint8_t> complement_q_scores_reversed(complement_read->qstring.begin(),
                                                      complement_read->qstring.end());
    std::reverse(complement_q_scores_reversed.begin(), complement_q_scores_reversed.end());

    std::vector<char> template_sequence(template_read->seq.begin(), template_read->seq.end());
    std::vector<uint8_t> template_q_scores(template_read->qstring.begin(),
                                           template_read->qstring.end());

    // Align the two reads to one another and print out the score.
    EdlibAlignResult result =
            edlibAlign(template_read->seq.data(), template_read->seq.size(),
                       complement_sequence_reverse_complement.data(),
                       complement_sequence_reverse_complement.size(), align_config);

    int query_cursor = 0;
    int target_cursor = result.startLocations[0];

    auto [alignment_start_end, cursors] = dorado::utils::get_trimmed_alignment(
            11, result.alignment, result.alignmentLength, target_cursor, query_cursor, 0,
            result.endLocations[0]);

    query_cursor = cursors.first;
    target_cursor = cursors.second;
    int start_alignment_position = alignment_start_end.first;
    int end_alignment_position = alignment_start_end.second;

    // TODO: perhaps its overkill having this function make this decision...
    int min_trimmed_alignment_length = 200;
    bool consensus_possible =
            (start_alignment_position < end_alignment_position) &&
            ((end_alignment_position - start_alignment_position) > min_trimmed_alignment_length);

    int stride = 5;  // TODO this needs to be passed in as a parameter

    if (consensus_possible) {
        // Move along the alignment, filling out the stereo-encoded tensor
        int max_size = template_read->raw_data.size(0) + complement_read->raw_data.size(0);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        int num_features = 13;
        auto tmp = torch::zeros({num_features, max_size}, opts);

        // Diagnostics one: Is sum of move vector the same length as the sequence
        int num_moves = 0;
        for (int i = 0; i < template_read->moves.size(); i++) {
            num_moves += template_read->moves[i];
        }

        int template_signal_cursor = 0;
        int complement_signal_cursor = 0;

        std::vector<uint8_t> template_moves_expanded;
        for (int i = 0; i < template_read->moves.size(); i++) {
            template_moves_expanded.push_back(template_read->moves[i]);
            for (int j = 0; j < stride - 1; j++) {
                template_moves_expanded.push_back(0);
            }
        }
        int extra_moves = template_moves_expanded.size() - template_read->raw_data.size(0);
        for (int i = 0; i < extra_moves; i++) {
            template_moves_expanded.pop_back();
        }

        int template_moves_seen = template_moves_expanded[template_signal_cursor];
        while (template_moves_seen < target_cursor + 1) {
            template_signal_cursor++;
            template_moves_seen += template_moves_expanded[template_signal_cursor];
        }

        std::vector<uint8_t> complement_moves_expanded;
        for (int i = 0; i < complement_read->moves.size(); i++) {
            complement_moves_expanded.push_back(complement_read->moves[i]);
            for (int j = 0; j < stride - 1; j++) {
                complement_moves_expanded.push_back(0);
            }
        }

        extra_moves = complement_moves_expanded.size() - complement_read->raw_data.size(0);
        for (int i = 0; i < extra_moves; i++) {
            complement_moves_expanded.pop_back();
        }
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

                // We are relying on strings of 0s ended in a 1.  It would be more efficient
                // in any case to store run length data above.
                // We're also assuming uint8_t is an alias for char (not guaranteed in principle).
                const auto* const start_ptr = &template_moves_expanded[template_signal_cursor];
                auto* const next_move_ptr =
                        static_cast<const uint8_t*>(std::memchr(start_ptr, 1, max_signal_length));
                const size_t sample_count =
                        next_move_ptr ? (next_move_ptr - start_ptr) : max_signal_length;

                float* const tmp_ptr = static_cast<float*>(tmp[0].data_ptr());
                const float* const raw_data_ptr =
                        static_cast<float*>(template_read->raw_data.data_ptr());
                // Assumes contiguity of successive elements.
                std::memcpy(&tmp_ptr[stereo_global_cursor + template_segment_length],
                            &raw_data_ptr[template_signal_cursor], sample_count * sizeof(float));

                template_signal_cursor += sample_count;
                template_segment_length += sample_count;
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

                // See comments above.
                const auto* const start_ptr = &complement_moves_expanded[complement_signal_cursor];
                auto* const next_move_ptr =
                        static_cast<const uint8_t*>(std::memchr(start_ptr, 1, max_signal_length));
                const size_t sample_count =
                        next_move_ptr ? (next_move_ptr - start_ptr) : max_signal_length;

                float* const tmp_ptr = static_cast<float*>(tmp[1].data_ptr());
                const float* const raw_data_ptr = static_cast<float*>(complement_signal.data_ptr());
                std::memcpy(&tmp_ptr[stereo_global_cursor + complement_segment_length],
                            &raw_data_ptr[complement_signal_cursor], sample_count * sizeof(float));

                complement_signal_cursor += sample_count;
                complement_segment_length += sample_count;
            }

            const int total_segment_length =
                    std::max(template_segment_length, complement_segment_length);
            const int start_ts = stereo_global_cursor;
            const int end_ts = start_ts + total_segment_length;

            // Now, add the nucleotides and q scores
            if (result.alignment[i] != 2) {
                const char nucleotide = template_sequence.at(target_cursor);
                const auto feature_idx = 2 + (0b11 & (nucleotide >> 2 ^ nucleotide >> 1));
                tmp.index_put_({feature_idx, Slice(start_ts, end_ts)}, 1.0f);
                tmp.index_put_({11, Slice(start_ts, end_ts)},
                               float(template_q_scores.at(target_cursor) - 33) / 90);
            }

            // Now, add the nucleotides and q scores
            if (result.alignment[i] != 1) {
                const char nucleotide = complement_sequence_reverse_complement.at(query_cursor);
                const auto feature_idx = 6 + (0b11 & (nucleotide >> 2 ^ nucleotide >> 1));
                tmp.index_put_({feature_idx, Slice(start_ts, end_ts)}, 1.0f);
                tmp.index_put_({12, Slice(start_ts, end_ts)},
                               float(complement_q_scores_reversed.at(query_cursor) - 33) / 90);
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

        tmp = tmp.index(
                {torch::indexing::Slice(None), torch::indexing::Slice(None, stereo_global_cursor)});

        read->read_id = template_read->read_id + ";" + complement_read->read_id;
        read->raw_data = tmp.to(torch::kFloat16);  // use the encoded signal
    }

    edlibFreeAlignResult(result);

    return read;
}
}  // namespace stereo_internal

namespace dorado {

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
        std::unique_lock<std::mutex> tc_lock(m_tc_map_mutex);

        if (m_template_complement_map.find(read->read_id) != m_template_complement_map.end()) {
            partner_id = m_template_complement_map[read->read_id];
            tc_lock.unlock();
            read_is_template = true;
            partner_found = true;
        } else {
            tc_lock.unlock();
            std::unique_lock<std::mutex> ct_lock(m_ct_map_mutex);
            if (m_complement_template_map.find(read->read_id) != m_complement_template_map.end()) {
                partner_id = m_complement_template_map[read->read_id];
                partner_found = true;
            }
            ct_lock.unlock();
        }

        if (partner_found) {
            std::unique_lock<std::mutex> read_cache_lock(m_read_cache_mutex);
            if (read_cache.find(partner_id) == read_cache.end()) {
                // Partner is not in the read cache
                read_cache[read->read_id] = read;
                read_cache_lock.unlock();
            } else {
                auto partner_read_itr = read_cache.find(partner_id);
                auto partner_read = partner_read_itr->second;
                read_cache.erase(partner_read_itr);
                read_cache_lock.unlock();

                std::shared_ptr<Read> template_read;
                std::shared_ptr<Read> complement_read;

                if (read_is_template) {
                    template_read = read;
                    complement_read = partner_read;
                } else {
                    complement_read = read;
                    template_read = partner_read;
                }

                std::shared_ptr<Read> stereo_encoded_read =
                        stereo_internal::stereo_encode(template_read, complement_read);

                if (stereo_encoded_read->raw_data.ndimension() ==
                    2) {  // 2 dims for stereo encoding, 1 for simplex
                    m_sink.push_read(
                            stereo_encoded_read);  // Strereo-encoded read created, send it to sink
                }
            }
        }
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(
        ReadSink& sink,
        std::map<std::string, std::string> template_complement_map)
        : ReadSink(1000), m_sink(sink), m_template_complement_map(template_complement_map) {
    // Set up the complement-template_map
    for (auto key : template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    int num_worker_threads = std::thread::hardware_concurrency();
    for (int i = 0; i < num_worker_threads; i++) {
        std::unique_ptr<std::thread> stereo_encoder_worker_thread =
                std::make_unique<std::thread>(&StereoDuplexEncoderNode::worker_thread, this);
        worker_threads.push_back(std::move(stereo_encoder_worker_thread));
    }
}

StereoDuplexEncoderNode::~StereoDuplexEncoderNode() {
    terminate();
    m_cv.notify_one();
    for (auto& t : worker_threads) {
        t->join();
    }
}

}  // namespace dorado
