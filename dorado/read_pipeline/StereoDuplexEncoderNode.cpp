#include "StereoDuplexEncoderNode.h"

#include "utils/duplex_utils.h"
#include "utils/sequence_utils.h"

#include <ATen/ATen.h>
#include <edlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace at::indexing;

namespace dorado {
DuplexReadPtr StereoDuplexEncoderNode::stereo_encode(const ReadPair& read_pair) {
    const ReadPair::ReadData& template_read = read_pair.template_read;
    const ReadPair::ReadData& complement_read = read_pair.complement_read;

    // We rely on the incoming read raw data being of type float16 to allow direct memcpy
    // of tensor elements.
    assert(template_read.read_common.raw_data.dtype() == at::kHalf);
    assert(complement_read.read_common.raw_data.dtype() == at::kHalf);

    assert(complement_read.read_common.attributes.mux == template_read.read_common.attributes.mux);
    assert(complement_read.read_common.attributes.channel_number ==
           template_read.read_common.attributes.channel_number);
    assert(complement_read.read_common.start_time_ms > template_read.read_common.start_time_ms);

    // We align the reverse complement of the complement read to the template read.
    const auto complement_sequence_reverse_complement =
            dorado::utils::reverse_complement(complement_read.read_common.seq);

    // Align the two reads to one another and print out the score.
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    auto temp_strand = template_read.read_common.seq.substr(
            template_read.seq_start, template_read.seq_end - template_read.seq_start);
    auto comp_strand = complement_sequence_reverse_complement.substr(
            complement_read.seq_start, complement_read.seq_end - complement_read.seq_start);

    EdlibAlignResult edlib_result =
            edlibAlign(temp_strand.data(), temp_strand.length(), comp_strand.data(),
                       comp_strand.length(), align_config);

    DuplexRead::StereoFeatureInputs stereo_feature_inputs;
    const auto alignment_size =
            static_cast<size_t>(edlib_result.endLocations[0] - edlib_result.startLocations[0]);
    stereo_feature_inputs.alignment.resize(alignment_size);
    std::memcpy(stereo_feature_inputs.alignment.data(), edlib_result.alignment, alignment_size);

    int target_cursor = template_read.seq_start;
    int query_cursor = complement_read.seq_start;

    int start_alignment_position = edlib_result.startLocations[0];
    int end_alignment_position = edlib_result.endLocations[0];
    spdlog::error("{} {} {}", start_alignment_position, end_alignment_position,
                  edlib_result.alignmentLength);

    edlibFreeAlignResult(edlib_result);

    // Edlib doesn't provide named constants for alignment array entries, so do it here.
    static constexpr unsigned char kAlignMatch = 0;
    static constexpr unsigned char kAlignInsertionToTarget = 1;
    static constexpr unsigned char kAlignInsertionToQuery = 2;
    static constexpr unsigned char kAlignMismatch = 3;

    // Move along the alignment, filling out the stereo-encoded tensor

    const auto opts = at::TensorOptions().dtype(at::ScalarType::Half).device(at::kCPU);

    static constexpr int kNumFeatures = 13;
    // Indices of features in the first dimension of the output tensor.
    static constexpr int kFeatureTemplateSignal = 0;
    static constexpr int kFeatureComplementSignal = 1;
    static constexpr int kFeatureTemplateFirstNucleotide = 2;
    static constexpr int kFeatureComplementFirstNucleotide = 6;
    static constexpr int kFeatureMoveTable = 10;
    static constexpr int kFeatureTemplateQScore = 11;
    static constexpr int kFeatureComplementQScore = 12;

    std::vector<uint8_t> template_moves_expanded;
    for (int i = 0; i < template_read.read_common.moves.size(); i++) {
        template_moves_expanded.push_back(template_read.read_common.moves[i]);
        for (int j = 0; j < m_input_signal_stride - 1; j++) {
            template_moves_expanded.push_back(0);
        }
    }

    int extra_padding =
            template_read.read_common.get_raw_data_samples() - template_moves_expanded.size();
    spdlog::error("extra_padding {}", extra_padding);
    for (int i = 0; i < extra_padding; i++) {
        template_moves_expanded.push_back(0);
    }

    std::vector<uint8_t> complement_moves_expanded;
    for (int i = 0; i < complement_read.read_common.moves.size(); i++) {
        complement_moves_expanded.push_back(complement_read.read_common.moves[i]);
        for (int j = 0; j < m_input_signal_stride - 1; j++) {
            complement_moves_expanded.push_back(0);
        }
    }

    extra_padding =
            complement_read.read_common.get_raw_data_samples() - complement_moves_expanded.size();
    for (int i = 0; i < extra_padding; i++) {
        complement_moves_expanded.push_back(0);
    }
    complement_moves_expanded.push_back(1);
    std::reverse(complement_moves_expanded.begin(), complement_moves_expanded.end());
    complement_moves_expanded.pop_back();

    int complement_signal_cursor = 0;
    int complement_moves_seen = complement_read.read_common.moves[0];
    while (complement_moves_seen < query_cursor + 1) {
        ++complement_signal_cursor;
        complement_moves_seen += complement_moves_expanded[complement_signal_cursor];
    }

    using SampleType = c10::Half;

    // libtorch indexing calls go on a carefree romp through various heap
    // allocations/deallocations and object constructions/destructions, and so are
    // glacially slow.  We therefore work with raw pointers within the main loop.
    const auto* const template_raw_data_ptr =
            template_read.read_common.raw_data.data_ptr<SampleType>();
    const auto complement_signal = at::flip(complement_read.read_common.raw_data, 0);
    const auto* const flipped_complement_raw_data_ptr = complement_signal.data_ptr<SampleType>();

    // Package the encoding generation function into a lambda so it can be called
    // in two modes -
    // 1. The mode without data copy is run to iterate through data structures
    // and determine the final size of the tensor needed to store the encoding.
    // This helps allocate the exact amount of data needed instead of overallocating
    // the buffer which helps bring down overall memory footprint.
    // 2. The mode with data copy that actually fills up the encoding tensor
    // with the right data needed for inference.
    auto generate_encoding = [&](std::optional<at::Tensor> tmp, int target_cursor, int query_cursor,
                                 int complement_signal_cursor) -> int {
        int template_signal_cursor = 0;

        int stereo_global_cursor = 0;  // Index into the stereo-encoded signal
        std::array<SampleType*, kNumFeatures> feature_ptrs;
        if (tmp) {
            for (int feature_idx = 0; feature_idx < kNumFeatures; ++feature_idx) {
                feature_ptrs[feature_idx] = (tmp.value())[feature_idx].data_ptr<SampleType>();
            }
        }
        for (auto alignment_entry : stereo_feature_inputs.alignment) {
            // We move along every alignment position. For every position we need to add signal and padding.

            // If there is *not* an insertion to the query, add the nucleotide from the target cursor
            int template_segment_length = 0;  // index into this segment in signal-space
            if (alignment_entry != kAlignInsertionToQuery) {
                const auto max_signal_length = complement_moves_expanded.size();
                const auto* const start_ptr = &template_moves_expanded[template_signal_cursor + 1];
                auto* const next_move_ptr =
                        static_cast<const uint8_t*>(std::memchr(start_ptr, 1, max_signal_length));
                const size_t sample_count =
                        next_move_ptr ? (next_move_ptr - start_ptr) : max_signal_length;

                if (tmp) {
                    // Assumes contiguity of successive elements.
                    std::memcpy(&feature_ptrs[kFeatureTemplateSignal][stereo_global_cursor],
                                &template_raw_data_ptr[template_signal_cursor],
                                (sample_count + 1) * sizeof(SampleType));
                }

                template_segment_length = sample_count + 1;
                template_signal_cursor += template_segment_length;
            }

            // If there is *not* an insertion to the target, add the nucleotide from the query cursor
            int complement_segment_length = 0;  // index into this segment in signal-space
            if (alignment_entry != kAlignInsertionToTarget) {
                const auto max_signal_length = complement_moves_expanded.size();
                const auto* const start_ptr =
                        &complement_moves_expanded[complement_signal_cursor + 1];
                auto* const next_move_ptr =
                        static_cast<const uint8_t*>(std::memchr(start_ptr, 1, max_signal_length));
                const size_t sample_count =
                        next_move_ptr ? (next_move_ptr - start_ptr) : max_signal_length;

                if (tmp) {
                    std::memcpy(&feature_ptrs[kFeatureComplementSignal][stereo_global_cursor],
                                &flipped_complement_raw_data_ptr[complement_signal_cursor],
                                (sample_count + 1) * sizeof(SampleType));
                }

                complement_segment_length = sample_count + 1;
                complement_signal_cursor += complement_segment_length;
            }

            const int total_segment_length =
                    std::max(template_segment_length, complement_segment_length);
            const int start_ts = stereo_global_cursor;

            // Converts Q scores from char to SampleType, with appropriate scale/offset.
            const auto convert_q_score = [](char q_in) {
                return static_cast<SampleType>(static_cast<float>(q_in - 33) / 90.0f);
            };

            // Now, add the nucleotides and q scores
            if (alignment_entry != kAlignInsertionToQuery) {
                if (tmp) {
                    const char nucleotide = template_read.read_common.seq[target_cursor];
                    const auto nucleotide_feature_idx = kFeatureTemplateFirstNucleotide +
                                                        dorado::utils::base_to_int(nucleotide);
                    std::fill_n(&feature_ptrs[nucleotide_feature_idx][start_ts],
                                total_segment_length, static_cast<SampleType>(1.0f));
                    std::fill_n(&feature_ptrs[kFeatureTemplateQScore][start_ts],
                                total_segment_length,
                                convert_q_score(template_read.read_common.qstring[target_cursor]));
                }

                // Anything but a query insertion causes the target cursor to advance.
                ++target_cursor;
            }

            // Now, add the nucleotides and q scores
            if (alignment_entry != kAlignInsertionToTarget) {
                if (tmp) {
                    const char nucleotide = complement_sequence_reverse_complement[query_cursor];
                    const auto nucleotide_feature_idx = kFeatureComplementFirstNucleotide +
                                                        dorado::utils::base_to_int(nucleotide);

                    std::fill_n(&feature_ptrs[nucleotide_feature_idx][start_ts],
                                total_segment_length, static_cast<SampleType>(1.0f));
                    std::fill_n(
                            &feature_ptrs[kFeatureComplementQScore][start_ts], total_segment_length,
                            convert_q_score(
                                    complement_read.read_common.qstring.rbegin()[query_cursor]));
                }

                // Anything but a target insertion causes the query cursor to advance.
                ++query_cursor;
            }

            if (tmp) {
                feature_ptrs[kFeatureMoveTable][stereo_global_cursor] =
                        static_cast<SampleType>(1);  // set the move table
            }

            // Update the global cursor
            stereo_global_cursor += total_segment_length;
        }
        return stereo_global_cursor;
    };

    // Call the encoding lambda first without data copy to get an estimate
    // of the encoding size.
    const auto encoding_tensor_size =
            generate_encoding(std::nullopt, target_cursor, query_cursor, complement_signal_cursor);

    const float pad_value =
            0.8 * std::min(at::min(complement_signal).item<float>(),
                           at::min(template_read.read_common.raw_data).item<float>());
    auto tmp = at::zeros({kNumFeatures, encoding_tensor_size}, opts);

    // Start with all signal feature entries equal to the padding value.
    tmp.index({at::indexing::Slice(None, 2)}) = pad_value;

    // Call the encoding lambda again, this time with the correctly sized tensor
    // allocated for the final data to be filled in.
    generate_encoding(tmp, target_cursor, query_cursor, complement_signal_cursor);

    auto read = std::make_unique<DuplexRead>();  // Return read
    read->read_common.read_id =
            template_read.read_common.read_id + ";" + complement_read.read_common.read_id;

    read->read_common.attributes.mux = template_read.read_common.attributes.mux;
    read->read_common.attributes.channel_number =
            template_read.read_common.attributes.channel_number;
    read->read_common.attributes.start_time = template_read.read_common.attributes.start_time;
    read->read_common.start_time_ms = template_read.read_common.start_time_ms;

    read->read_common.read_tag = template_read.read_common.read_tag;
    read->read_common.client_id = template_read.read_common.client_id;
    read->read_common.raw_data = tmp;  // use the encoded signal
    read->read_common.is_duplex = true;
    read->read_common.run_id = template_read.read_common.run_id;
    read->read_common.flowcell_id = template_read.read_common.flowcell_id;
    read->read_common.position_id = template_read.read_common.position_id;
    read->read_common.experiment_id = template_read.read_common.experiment_id;

    //auto& stereo_feature_inputs = read->stereo_feature_inputs;

    m_num_encoded_pairs++;

    return read;
}

void StereoDuplexEncoderNode::worker_thread() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<ReadPair>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read_pair = std::get<ReadPair>(std::move(message));
        auto stereo_encoded_read = stereo_encode(read_pair);

        send_message_to_sink(
                std::move(stereo_encoded_read));  // Stereo-encoded read created, send it to sink
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(int input_signal_stride)
        : MessageSink(1000), m_input_signal_stride(input_signal_stride) {
    start_threads();
}

void StereoDuplexEncoderNode::start_threads() {
    const int num_worker_threads = std::thread::hardware_concurrency();
    for (int i = 0; i < num_worker_threads; ++i) {
        std::unique_ptr<std::thread> stereo_encoder_worker_thread =
                std::make_unique<std::thread>(&StereoDuplexEncoderNode::worker_thread, this);
        m_worker_threads.push_back(std::move(stereo_encoder_worker_thread));
    }
}

void StereoDuplexEncoderNode::terminate_impl() {
    terminate_input_queue();
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();
}

void StereoDuplexEncoderNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats StereoDuplexEncoderNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    stats["encoded_pairs"] = m_num_encoded_pairs;
    return stats;
}

}  // namespace dorado
