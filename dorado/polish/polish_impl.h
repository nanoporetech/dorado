#pragma once

#include "polish/architectures/decoder_factory.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_factory.h"
#include "polish/consensus_result.h"
#include "polish/sample.h"
#include "polish/trim.h"
#include "polish/window.h"
#include "utils/span.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

/**
 * \brief Struct which holds data prepared for inference. In practice,
 *          vectors here hold one batch for inference. Both vectors should
 *          have identical length.
 */
struct InferenceData {
    std::vector<Sample> samples;
    std::vector<TrimInfo> trims;
};

/**
 * \brief Struct which holds output of inference, passed into the decoding thread.
 */
struct DecodeData {
    std::vector<Sample> samples;
    torch::Tensor logits;
    std::vector<TrimInfo> trims;
};

/**
 * \brief For a given consensus, goes through the sequence and removes all '*' characters.
 *          It also removes the corresponding positions from the quality field.
 *          Works in-place.
 */
void remove_deletions(ConsensusResult& cons);

/**
 * \brief Takes consensus results for all samples and stitches them into full sequences.
 *          If fill_gaps is true, missing pieces will be filled with either the draft sequence
 *          or with an optional fill_char character.
 */
std::vector<ConsensusResult> stitch_sequence(
        const std::filesystem::path& in_draft_fn,
        const std::string& header,
        const std::vector<ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char);

/**
 * \brief This function performs the following operations:
 *          1. Merges adjacent samples, which were split for efficiency of computing the pileup.
 *          2. Checks for discontinuities in any of the samples (based on major positions) and splits them.
 *          3. Splits the merged samples into equally sized pieces which will be used for inference to prevent memory usage spikes.
 * \param window_samples Input samples which will be merged and split. Non-const to enable moving of data.
 * \param encoder Encoder used to produce the sample tensors. It is needed becaue of the EncoderBase::merge_adjacent_samples() function.
 * \param bam_regions BAM region coordinates. This is a Span to facilitate batching of BAM regions from the outside.
 * \param bam_region_intervals Range of IDs of window_samples which comprise this BAM region. E.g. BAM region 0 uses window_samples[0:5], BAM region 1 uses window_samples[5:9], etc.
 *                              This is a Span to facilitate batching of BAM regions from the outside and avoid copying vectors.
 * \param num_threads Number of threads for procesing.
 * \param window_len Length of the window to split the final samples into.
 * \param window_overlap Overlap between neighboring windows when splitting.
 * \param window_interval_offset Used for batching bam_region_intervals, because window_samples.size() matches the total size of bam_region_intervals,
 *                                  while coordinates of each BAM region interval are global and produced before draft batching on the client side.
 */
std::pair<std::vector<Sample>, std::vector<TrimInfo>> merge_and_split_bam_regions_in_parallel(
        std::vector<Sample>& window_samples,
        const EncoderBase& encoder,
        const Span<const Window> bam_regions,
        const Span<const Interval> bam_region_intervals,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t window_interval_offset);

/**
 * \brief For each input window (region of the draft) runs the given encoder and produces a sample.
 *          The BamFile handels are used to fetch the pileup data and encode regions.
 *          Encoding is parallelized, where the actual number of threads is min(bam_handles.size(), num_threads, windows.size()).
 */
std::vector<Sample> encode_windows_in_parallel(
        std::vector<BamFile>& bam_handles,
        const EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const Window> windows,
        const int32_t num_threads);

/**
 * \brief Creates windows from given input draft sequences or regions. If regions vector is empty, it will split all
 *          input draft sequences into windows.
 */
std::vector<Window> create_bam_regions(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t bam_chunk_len,
        const int32_t window_overlap,
        const std::vector<std::string>& regions);

}  // namespace dorado::polisher
