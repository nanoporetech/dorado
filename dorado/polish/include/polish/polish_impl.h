#pragma once

#include "secondary/architectures/model_config.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/common/bam_file.h"
#include "secondary/common/interval.h"
#include "secondary/common/stats.h"
#include "secondary/common/variant.h"
#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/consensus/window.h"
#include "secondary/features/decoder_factory.h"
#include "secondary/features/encoder_factory.h"
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
    std::vector<std::unique_ptr<secondary::EncoderBase>> encoders;
    std::unique_ptr<secondary::DecoderBase> decoder;
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

struct WorkerReturnStatus {
    bool exception_thrown{false};
    std::string message;
};

/**
 * \brief Creates all resources required to run polishing.
 */
PolisherResources create_resources(const secondary::ModelConfig& model_config,
                                   const std::filesystem::path& in_ref_fn,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   int32_t num_bam_threads,
                                   int32_t num_inference_threads,
                                   bool full_precision,
                                   const std::string& read_group,
                                   const std::string& tag_name,
                                   int32_t tag_value,
                                   const std::optional<bool>& tag_keep_missing_override,
                                   const std::optional<int32_t>& min_mapq_override,
                                   const std::optional<secondary::HaplotagSource>& haptag_source,
                                   const std::optional<std::filesystem::path>& phasing_bin_fn);

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
 * \param sample_results is a 2D vector of: [window x haplotype]. For haploid sequences, inner
 *                          vector should be of size 1.
 * \returns a 2D vector of dimensions: [part x haplotype], where parts are gap-separated portions
 *          of the consensus.
 */
std::vector<std::vector<secondary::ConsensusResult>> stitch_sequence(
        const hts_io::FastxRandomReader& fastx_reader,
        const std::string& header,
        const std::vector<std::vector<secondary::ConsensusResult>>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        bool fill_gaps,
        const std::optional<char>& fill_char);

/**
 * \brief Creates windows from given input draft sequences or regions. If regions vector is empty, it will split all
 *          input draft sequences into windows.
 */
std::vector<secondary::Window> create_windows_from_regions(
        const std::vector<secondary::Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        int32_t bam_chunk_len,
        int32_t window_overlap);

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
void decode_samples_in_parallel(std::vector<std::vector<secondary::ConsensusResult>>& results_cons,
                                std::vector<secondary::VariantCallingSample>& results_vc_data,
                                utils::AsyncQueue<DecodeData>& decode_queue,
                                secondary::Stats& stats,
                                std::atomic<bool>& worker_terminate,
                                polisher::WorkerReturnStatus& ret_status,
                                const secondary::DecoderBase& decoder,
                                int32_t num_threads,
                                int32_t min_depth,
                                bool collect_vc_data,
                                bool continue_on_exception);

void infer_samples_in_parallel(utils::AsyncQueue<InferenceData>& batch_queue,
                               utils::AsyncQueue<DecodeData>& decode_queue,
                               std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
                               std::atomic<bool>& worker_terminate,
                               const std::vector<c10::optional<c10::Stream>>& streams,
                               const std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
                               const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                               bool continue_on_exception);

void sample_producer(PolisherResources& resources,
                     const std::vector<secondary::Window>& bam_regions,
                     const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                     int32_t num_threads,
                     int32_t batch_size,
                     int32_t window_len,
                     int32_t window_overlap,
                     int32_t bam_subchunk_len,
                     bool continue_on_exception,
                     utils::AsyncQueue<InferenceData>& infer_data,
                     std::atomic<bool>& worker_terminate,
                     WorkerReturnStatus& ret_status);

/// \brief Dimensions: [draft_id x part_id x haplotype_id]
std::vector<std::vector<std::vector<secondary::ConsensusResult>>> construct_consensus_seqs(
        const secondary::Interval& region_batch,
        const std::vector<std::vector<secondary::ConsensusResult>>& all_results_cons,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        bool fill_gaps,
        const std::optional<char>& fill_char,
        hts_io::FastxRandomReader& draft_reader);

// Explicit full qualification of the Interval so it is not confused with the one from the IntervalTree library.
std::vector<secondary::Variant> call_variants(
        std::atomic<bool>& worker_terminate,
        secondary::Stats& stats,
        const secondary::Interval& region_batch,
        const std::vector<secondary::VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        bool ambig_ref,
        bool gvcf,
        int32_t num_threads,
        bool continue_on_exception);

}  // namespace dorado::polisher
