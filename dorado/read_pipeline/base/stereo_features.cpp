#include "stereo_features.h"

#include "utils/sequence_utils.h"

#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

using namespace at::indexing;

namespace dorado {

at::Tensor generate_stereo_features(const DuplexRead::StereoFeatureInputs& feature_inputs) {
    const int target_cursor = static_cast<int>(feature_inputs.template_seq_start);
    const int query_cursor = static_cast<int>(feature_inputs.complement_seq_start);

    // Edlib doesn't provide named constants for alignment array entries, so do it here.
    // static constexpr unsigned char kAlignMatch = 0;
    static constexpr unsigned char kAlignInsertionToTarget = 1;
    static constexpr unsigned char kAlignInsertionToQuery = 2;
    // static constexpr unsigned char kAlignMismatch = 3;

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

    int template_signal_cursor = 0;
    int complement_signal_cursor = 0;

    std::vector<uint8_t> template_moves_expanded;
    for (size_t i = 0; i < feature_inputs.template_moves.size(); ++i) {
        template_moves_expanded.push_back(feature_inputs.template_moves[i]);
        for (int j = 0; j < feature_inputs.signal_stride - 1; ++j) {
            template_moves_expanded.push_back(0);
        }
    }

    size_t extra_padding = feature_inputs.template_signal.size(0) - template_moves_expanded.size();
    for (size_t i = 0; i < extra_padding; ++i) {
        template_moves_expanded.push_back(0);
    }

    int template_moves_seen = template_moves_expanded[template_signal_cursor];
    while (template_moves_seen < target_cursor + 1) {
        ++template_signal_cursor;
        template_moves_seen += template_moves_expanded[template_signal_cursor];
    }

    std::vector<uint8_t> complement_moves_expanded;
    for (size_t i = 0; i < feature_inputs.complement_moves.size(); ++i) {
        complement_moves_expanded.push_back(feature_inputs.complement_moves[i]);
        for (int j = 0; j < feature_inputs.signal_stride - 1; ++j) {
            complement_moves_expanded.push_back(0);
        }
    }

    extra_padding = feature_inputs.complement_signal.size(0) - complement_moves_expanded.size();
    for (size_t i = 0; i < extra_padding; ++i) {
        complement_moves_expanded.push_back(0);
    }
    complement_moves_expanded.push_back(1);
    std::reverse(complement_moves_expanded.begin(), complement_moves_expanded.end());
    complement_moves_expanded.pop_back();

    int complement_moves_seen = feature_inputs.complement_moves[complement_signal_cursor];
    while (complement_moves_seen < query_cursor + 1) {
        ++complement_signal_cursor;
        complement_moves_seen += complement_moves_expanded[complement_signal_cursor];
    }

    using SampleType = c10::Half;

    // libtorch indexing calls go on a carefree romp through various heap
    // allocations/deallocations and object constructions/destructions, and so are
    // glacially slow.  We therefore work with raw pointers within the main loop.
    const auto* const template_raw_data_ptr = feature_inputs.template_signal.data_ptr<SampleType>();
    const auto* const flipped_complement_raw_data_ptr =
            feature_inputs.complement_signal.data_ptr<SampleType>();

    // Package the encoding generation function into a lambda so it can be called
    // in two modes -
    // 1. The mode without data copy is run to iterate through data structures
    // and determine the final size of the tensor needed to store the encoding.
    // This helps allocate the exact amount of data needed instead of overallocating
    // the buffer which helps bring down overall memory footprint.
    // 2. The mode with data copy that actually fills up the encoding tensor
    // with the right data needed for inference.
    auto determine_encoding = [&](at::Tensor* stereo_features, int current_target_cursor,
                                  int current_query_cursor, int current_template_signal_cursor,
                                  int current_complement_signal_cursor) -> int {
        size_t stereo_global_cursor = 0;  // Index into the stereo-encoded signal
        std::array<SampleType*, kNumFeatures> feature_ptrs;
        if (stereo_features) {
            for (int feature_idx = 0; feature_idx < kNumFeatures; ++feature_idx) {
                feature_ptrs[feature_idx] = (*stereo_features)[feature_idx].data_ptr<SampleType>();
            }
        }
        for (auto alignment_entry : feature_inputs.alignment) {
            // We move along every alignment position. For every position we need to add signal and padding.
            size_t total_segment_length = 0;

            // Adds the segment of the signal associated with the current base, updating
            // total_segment_length to reflect the maximum across successive invocations.
            auto add_signal = [&total_segment_length, stereo_global_cursor, &stereo_features,
                               feature_ptrs](const std::vector<uint8_t>& moves_expanded,
                                             int& signal_cursor, int feature_index,
                                             const SampleType* const raw_data_ptr) {
                const auto max_signal_length = moves_expanded.size();
                const auto* const start_ptr = &moves_expanded[signal_cursor + 1];
                const auto* const next_move_ptr =
                        static_cast<const uint8_t*>(std::memchr(start_ptr, 1, max_signal_length));
                const size_t sample_count =
                        next_move_ptr ? (next_move_ptr - start_ptr) : max_signal_length;

                if (stereo_features) {
                    // Assumes contiguity of successive elements.
                    std::memcpy(&feature_ptrs[feature_index][stereo_global_cursor],
                                &raw_data_ptr[signal_cursor],
                                (sample_count + 1) * sizeof(SampleType));
                }

                const size_t segment_length = sample_count + 1;
                total_segment_length = std::max(total_segment_length, segment_length);
                signal_cursor += static_cast<int>(segment_length);
            };

            // If there is *not* an insertion to the query, add the nucleotide from the target cursor.
            if (alignment_entry != kAlignInsertionToQuery) {
                add_signal(template_moves_expanded, current_template_signal_cursor,
                           kFeatureTemplateSignal, template_raw_data_ptr);
            }

            // If there is *not* an insertion to the target, add the nucleotide from the query cursor
            if (alignment_entry != kAlignInsertionToTarget) {
                add_signal(complement_moves_expanded, current_complement_signal_cursor,
                           kFeatureComplementSignal, flipped_complement_raw_data_ptr);
            }

            // Now, add the nucleotides and q scores.  We need to do this after determining
            // total_segment_length.
            auto add_nucleotide_and_q = [total_segment_length, stereo_global_cursor, feature_ptrs](
                                                const char nucleotide, const char q_score,
                                                const int first_nucleotide_feature_index,
                                                const int q_feature_index) {
                const auto nucleotide_feature_idx =
                        first_nucleotide_feature_index + dorado::utils::base_to_int(nucleotide);
                std::fill_n(&feature_ptrs[nucleotide_feature_idx][stereo_global_cursor],
                            total_segment_length, static_cast<SampleType>(1.0f));

                // Convert Q scores from char to SampleType, with appropriate scale/offset.
                const auto q_score_sample_type =
                        static_cast<SampleType>(static_cast<float>(q_score - 33) / 90.0f);
                std::fill_n(&feature_ptrs[q_feature_index][stereo_global_cursor],
                            total_segment_length, q_score_sample_type);
            };

            if (alignment_entry != kAlignInsertionToQuery) {
                if (stereo_features) {
                    add_nucleotide_and_q(feature_inputs.template_seq[current_target_cursor],
                                         feature_inputs.template_qstring[current_target_cursor],
                                         kFeatureTemplateFirstNucleotide, kFeatureTemplateQScore);
                }

                // Anything but a query insertion causes the target cursor to advance.
                ++current_target_cursor;
            }

            // Now, add the nucleotides and q scores
            if (alignment_entry != kAlignInsertionToTarget) {
                if (stereo_features) {
                    add_nucleotide_and_q(
                            feature_inputs.complement_seq[current_query_cursor],
                            feature_inputs.complement_qstring.rbegin()[current_query_cursor],
                            kFeatureComplementFirstNucleotide, kFeatureComplementQScore);
                }

                // Anything but a target insertion causes the query cursor to advance.
                ++current_query_cursor;
            }

            if (stereo_features) {
                feature_ptrs[kFeatureMoveTable][stereo_global_cursor] =
                        static_cast<SampleType>(1);  // set the move table
            }

            // Update the global cursor
            stereo_global_cursor += total_segment_length;
        }
        return static_cast<int>(stereo_global_cursor);
    };

    // Call the encoding lambda first without data copy to get an estimate
    // of the encoding size.
    const auto encoding_tensor_size = determine_encoding(
            nullptr, target_cursor, query_cursor, template_signal_cursor, complement_signal_cursor);

    const float pad_value = 0.8f * std::min(at::min(feature_inputs.complement_signal).item<float>(),
                                            at::min(feature_inputs.template_signal).item<float>());
    auto stereo_features = at::zeros({kNumFeatures, encoding_tensor_size}, opts);

    // Start with all signal feature entries equal to the padding value.
    stereo_features.index({at::indexing::Slice(None, 2)}) = pad_value;

    // Call the encoding lambda again, this time with the correctly sized tensor
    // allocated for the final data to be filled in.
    determine_encoding(&stereo_features, target_cursor, query_cursor, template_signal_cursor,
                       complement_signal_cursor);

    return stereo_features;
}

}  // namespace dorado
