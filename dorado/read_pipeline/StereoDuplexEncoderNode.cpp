#include "StereoDuplexEncoderNode.h"

#include "torch_utils/duplex_utils.h"
#include "utils/sequence_utils.h"

#include <ATen/Functions.h>
#include <edlib.h>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace dorado {

DuplexReadPtr StereoDuplexEncoderNode::stereo_encode(ReadPair read_pair) {
    ReadPair::ReadData& template_read = read_pair.template_read;
    ReadPair::ReadData& complement_read = read_pair.complement_read;

    // We rely on the incoming read raw data being of type float16 to allow direct memcpy
    // of tensor elements.
    assert(template_read.read_common.raw_data.dtype() == at::kHalf);
    assert(complement_read.read_common.raw_data.dtype() == at::kHalf);

    assert(complement_read.read_common.attributes.mux == template_read.read_common.attributes.mux);
    assert(complement_read.read_common.attributes.channel_number ==
           template_read.read_common.attributes.channel_number);
    assert(complement_read.read_common.start_time_ms > template_read.read_common.start_time_ms);

    // We align the reverse complement of the complement read to the template read.
    auto complement_sequence_reverse_complement =
            dorado::utils::reverse_complement(complement_read.read_common.seq);

    // Align the two reads to one another and print out the score.
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    auto temp_strand = std::string_view{template_read.read_common.seq}.substr(
            template_read.seq_start, template_read.seq_end - template_read.seq_start);
    auto comp_strand = std::string_view{complement_sequence_reverse_complement}.substr(
            complement_read.seq_start, complement_read.seq_end - complement_read.seq_start);

    EdlibAlignResult edlib_result =
            edlibAlign(temp_strand.data(), static_cast<int>(temp_strand.length()),
                       comp_strand.data(), static_cast<int>(comp_strand.length()), align_config);

    // Store the alignment result, along with other inputs necessary for generating the stereo input
    // features, in DuplexRead.
    auto read = std::make_unique<DuplexRead>();
    DuplexRead::StereoFeatureInputs& stereo_feature_inputs = read->stereo_feature_inputs;
    stereo_feature_inputs.signal_stride = m_input_signal_stride;

    const auto alignment_size =
            static_cast<size_t>(edlib_result.endLocations[0] - edlib_result.startLocations[0]);
    stereo_feature_inputs.alignment.resize(alignment_size);
    std::memcpy(stereo_feature_inputs.alignment.data(),
                &edlib_result.alignment[edlib_result.startLocations[0]], alignment_size);
    edlibFreeAlignResult(edlib_result);

    stereo_feature_inputs.template_seq_start = template_read.seq_start;
    stereo_feature_inputs.template_seq = std::move(template_read.read_common.seq);
    stereo_feature_inputs.template_qstring = std::move(template_read.read_common.qstring);
    stereo_feature_inputs.template_moves = std::move(template_read.read_common.moves);
    stereo_feature_inputs.template_signal = std::move(template_read.read_common.raw_data);

    stereo_feature_inputs.complement_seq_start = complement_read.seq_start;
    stereo_feature_inputs.complement_seq = std::move(complement_sequence_reverse_complement);
    stereo_feature_inputs.complement_qstring = std::move(complement_read.read_common.qstring);
    stereo_feature_inputs.complement_moves = std::move(complement_read.read_common.moves);
    stereo_feature_inputs.complement_signal = at::flip(complement_read.read_common.raw_data, 0);

    read->read_common.read_id =
            template_read.read_common.read_id + ";" + complement_read.read_common.read_id;

    read->read_common.attributes.mux = template_read.read_common.attributes.mux;
    read->read_common.attributes.channel_number =
            template_read.read_common.attributes.channel_number;
    read->read_common.attributes.start_time = template_read.read_common.attributes.start_time;
    read->read_common.start_time_ms = template_read.read_common.start_time_ms;

    read->read_common.read_tag = template_read.read_common.read_tag;
    read->read_common.client_info = std::move(template_read.read_common.client_info);
    read->read_common.is_duplex = true;
    read->read_common.run_id = std::move(template_read.read_common.run_id);
    read->read_common.flowcell_id = std::move(template_read.read_common.flowcell_id);
    read->read_common.position_id = std::move(template_read.read_common.position_id);
    read->read_common.experiment_id = std::move(template_read.read_common.experiment_id);

    ++m_num_encoded_pairs;

    return read;
}

void StereoDuplexEncoderNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<ReadPair>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read_pair = std::get<ReadPair>(std::move(message));
        auto stereo_encoded_read = stereo_encode(std::move(read_pair));

        send_message_to_sink(
                std::move(stereo_encoded_read));  // Stereo-encoded read created, send it to sink
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(int input_signal_stride)
        : MessageSink(1000, std::thread::hardware_concurrency()),
          m_input_signal_stride(input_signal_stride) {}

stats::NamedStats StereoDuplexEncoderNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    stats["encoded_pairs"] = static_cast<double>(m_num_encoded_pairs);
    return stats;
}

}  // namespace dorado
