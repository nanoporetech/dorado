#pragma once

#include "polish_stats.h"
#include "secondary/architectures/model_config.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/bam_file.h"
#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/consensus/window.h"
#include "secondary/features/decoder_factory.h"
#include "secondary/features/encoder_factory.h"
#include "secondary/interval.h"
#include "secondary/variant.h"
#include "utils/AsyncQueue.h"
#include "utils/span.h"
#include "utils/stats.h"
#include "utils/timer_high_res.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Forward declare the FastxRandomReader.
namespace dorado::hts_io {
class FastxRandomReader;
}  // namespace dorado::hts_io

namespace dorado::polisher {

enum class DeviceType { CPU, CUDA, METAL, UNKNOWN };

struct DeviceInfo {
    std::string name;
    DeviceType type;
    torch::Device device;
};

struct PolisherResources {
    std::unique_ptr<secondary::EncoderBase> encoder;
    std::unique_ptr<secondary::DecoderBase> decoder;
    std::vector<secondary::BamFile> bam_handles;
    std::vector<DeviceInfo> devices;
    std::vector<std::shared_ptr<secondary::ModelTorchBase>> models;
    std::vector<c10::optional<c10::Stream>> streams;
};

/**
 * \brief Struct which holds data prepared for inference. In practice,
 *          vectors here hold one batch for inference. Both vectors should
 *          have identical length.
 */
struct InferenceData {
    std::vector<secondary::Sample> samples;
    std::vector<secondary::TrimInfo> trims;
};

/**
 * \brief Struct which holds output of inference, passed into the decoding thread.
 */
struct DecodeData {
    std::vector<secondary::Sample> samples;
    torch::Tensor logits;
    std::vector<secondary::TrimInfo> trims;
};

/**
 * \brief Creates all resources required to run polishing.
 */
PolisherResources create_resources(const secondary::ModelConfig& model_config,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   const int32_t num_bam_threads,
                                   const int32_t num_inference_threads,
                                   const bool full_precision,
                                   const std::string& read_group,
                                   const std::string& tag_name,
                                   const int32_t tag_value,
                                   const std::optional<bool>& tag_keep_missing_override,
                                   const std::optional<int32_t>& min_mapq_override);

/**
 * \brief For a given consensus, goes through the sequence and removes all '*' characters.
 *          It also removes the corresponding positions from the quality field.
 *          Works in-place.
 */
void remove_deletions(secondary::ConsensusResult& cons);

/**
 * \brief Takes consensus results for all samples and stitches them into full sequences.
 *          If fill_gaps is true, missing pieces will be filled with either the draft sequence
 *          or with an optional fill_char character.
 */
std::vector<secondary::ConsensusResult> stitch_sequence(
        const hts_io::FastxRandomReader& fastx_reader,
        const std::string& header,
        const std::vector<secondary::ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char);

/**
 * \brief This function performs the following operations:
 *          1. Merges adjacent samples, which were split for efficiency of computing the pileup.
 *          2. Checks for discontinuities in any of the samples (based on major positions) and splits them.
 *          3. Splits the merged samples into equally sized pieces which will be used for inference to prevent memory usage spikes.
 * \param window_samples Input samples which will be merged and split. Non-const to enable moving of data.
 * \param encoder Encoder used to produce the sample tensors. It is needed becaue of the secondary::EncoderBase::merge_adjacent_samples() function.
 * \param bam_regions BAM region coordinates. This is a Span to facilitate batching of BAM regions from the outside.
 * \param bam_region_intervals Range of IDs of window_samples which comprise this BAM region. E.g. BAM region 0 uses window_samples[0:5], BAM region 1 uses window_samples[5:9], etc.
 *                              This is a Span to facilitate batching of BAM regions from the outside and avoid copying vectors.
 * \param num_threads Number of threads for procesing.
 * \param window_len Length of the window to split the final samples into.
 * \param window_overlap Overlap between neighboring windows when splitting.
 * \param window_interval_offset Used for batching bam_region_intervals, because window_samples.size() matches the total size of bam_region_intervals,
 *                                  while coordinates of each BAM region interval are global and produced before draft batching on the client side.
 */
std::pair<std::vector<secondary::Sample>, std::vector<secondary::TrimInfo>>
merge_and_split_bam_regions_in_parallel(std::vector<secondary::Sample>& window_samples,
                                        const secondary::EncoderBase& encoder,
                                        const Span<const secondary::Window> bam_regions,
                                        const Span<const secondary::Interval> bam_region_intervals,
                                        const int32_t num_threads,
                                        const int32_t window_len,
                                        const int32_t window_overlap,
                                        const int32_t window_interval_offset);

/**
 * \brief For each input window (region of the draft) runs the given encoder and produces a sample.
 *          The BamFile handels are used to fetch the pileup data and encode regions.
 *          Encoding is parallelized, where the actual number of threads is min(bam_handles.size(), num_threads, windows.size()).
 */
std::vector<secondary::Sample> encode_windows_in_parallel(
        std::vector<secondary::BamFile>& bam_handles,
        const secondary::EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const secondary::Window> windows,
        const int32_t num_threads);

/**
 * \brief Creates windows from given input draft sequences or regions. If regions vector is empty, it will split all
 *          input draft sequences into windows.
 */
std::vector<secondary::Window> create_windows_from_regions(
        const std::vector<secondary::Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap);

/**
 * \brief Fetches the decode data from an async queue, decodes the consensus and collects
 *          the consensus results. It also returns a vector of the decode data taken off of the queue
 *          (i.e. the input used for decoding). This will be needed downstream for variant calling.
 * \param results Return vector of consensus results.
 * \param decode_data Return vector of input data used for decoding, taken from the queue.
 * \param decode_queue Queue where messages will be received.
 * \param polish_stats Stats object, for the progress bar.
 * \param decoder Decoder to convert integers to bases.
 * \param num_threads Number of threads for processing.
 * \param min_depth Consensus sequences will be split in regions of insufficient depth.
 */
void decode_samples_in_parallel(std::vector<secondary::ConsensusResult>& results_cons,
                                std::vector<secondary::VariantCallingSample>& results_vc_data,
                                utils::AsyncQueue<DecodeData>& decode_queue,
                                PolishStats& polish_stats,
                                const secondary::DecoderBase& decoder,
                                const int32_t num_threads,
                                const int32_t min_depth,
                                const bool collect_vc_data);

void infer_samples_in_parallel(utils::AsyncQueue<InferenceData>& batch_queue,
                               utils::AsyncQueue<DecodeData>& decode_queue,
                               std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
                               const std::vector<c10::optional<c10::Stream>>& streams,
                               const secondary::EncoderBase& encoder);

void sample_producer(PolisherResources& resources,
                     const std::vector<secondary::Window>& bam_regions,
                     const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                     const int32_t num_threads,
                     const int32_t batch_size,
                     const int32_t window_len,
                     const int32_t window_overlap,
                     const int32_t bam_subchunk_len,
                     utils::AsyncQueue<InferenceData>& infer_data);

std::vector<std::vector<secondary::ConsensusResult>> construct_consensus_seqs(
        const secondary::Interval& region_batch,
        const std::vector<secondary::ConsensusResult>& all_results_cons,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool fill_gaps,
        const std::optional<char>& fill_char,
        hts_io::FastxRandomReader& draft_reader);

// Explicit full qualification of the Interval so it is not confused with the one from the IntervalTree library.
std::vector<secondary::Variant> call_variants(
        const secondary::Interval& region_batch,
        const std::vector<secondary::VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        PolishStats& polish_stats);

}  // namespace dorado::polisher
