#pragma once

#include "hts_utils/FastxRandomReader.h"
#include "secondary/architectures/model_config.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/common/batched_data.h"
#include "secondary/common/interval.h"
#include "secondary/common/interval_tree_types.h"
#include "secondary/common/region.h"
#include "secondary/common/stats.h"
#include "secondary/common/variant.h"
#include "secondary/common/vcf_writer.h"
#include "secondary/common/window.h"
#include "secondary/common/worker_return_status.h"
#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/sample.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_factory.h"
#include "secondary/features/encoder_factory.h"
#include "secondary/features/kadayashi_options.h"
#include "secondary/features/variant_candidate_source.h"
#include "secondary/kadayashi/local_haplotagging.h"
#include "utils/AsyncQueue.h"
#include "variant/decode_data.h"
#include "variant/inference_data.h"
#include "variant/variant_resources.h"

#include <IntervalTree.h>

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::smallvar {

// clang-format off
struct ChromosomeReduceData {
    std::mutex mtx;
    int64_t seq_id{0};
    std::string seq_name{};
    int64_t seq_len{0};
    int64_t progress_target{0};     // Number of selected unique input bases for progress tracking.
    bool ready{false};      // Is this object ready? Num_items is incremented dynamically, and last processed BAM region for each chromosome will mark this as ready.
    int64_t num_bam_regions{0};
    int64_t remaining_bam_regions{0};   // How many bam regions still need to be processed before the "ready" flag is turned true.
    int64_t num_samples{0};
    std::vector<secondary::Variant> variants_inference;
    std::vector<secondary::Variant> variants_simple;
    std::vector<secondary::Variant> variants_merged;
    std::vector<secondary::RegionInt> selected_regions;
    std::vector<secondary::IntervalInt64> processed_regions;
    std::vector<std::pair<uint32_t, uint32_t>> read_endpoints;
};
// clang-format on

/**
 * \brief Sets the worker_terminate flag to true and notifies all.
 * \param worker_terminate Shared termination flag used by all async variant workers.
 */
void signal_worker_terminate(std::atomic<bool>& worker_terminate);

/**
 * \brief Creates all resources required to run variant calling.
 * \param model_config Secondary model configuration used to construct the encoder, decoder and
 *                     model instances.
 * \param in_ref_fn Path to the draft/reference FASTX file.
 * \param in_aln_bam_fn Path to the aligned BAM input.
 * \param device_str Device specifier used for inference.
 * \param num_bam_threads Number of threads used by the alignment reader and encoder path.
 * \param num_inference_threads Number of inference worker threads to create.
 * \param full_precision Whether the model should run in full precision mode.
 * \param read_group Optional read group filter.
 * \param tag_name Alignment tag name used for filtering.
 * \param tag_value Alignment tag value used for filtering.
 * \param min_snp_accuracy Minimum SNP accuracy threshold passed to Kadayashi.
 * \param tag_keep_missing_override Optional override for retaining reads that do not carry the tag.
 * \param min_mapq_override Optional override for the minimum mapping quality filter.
 * \param haptag_source Optional haplotag source selection.
 * \param phasing_bin_fn Optional path to an external phasing binary.
 * \param kadayashi_opt Kadayashi-specific configuration.
 * \param legacy_feature_gen Use legacy read alignment feature generation.
 * \return Fully initialized resources for the async variant-calling pipeline.
 */
VariantResources create_resources(const secondary::ModelConfig& model_config,
                                  const std::filesystem::path& in_ref_fn,
                                  const std::filesystem::path& in_aln_bam_fn,
                                  const std::string& device_str,
                                  int32_t num_bam_threads,
                                  int32_t num_inference_threads,
                                  bool full_precision,
                                  const std::string& read_group,
                                  const std::string& tag_name,
                                  int32_t tag_value,
                                  double min_snp_accuracy,
                                  const std::optional<bool>& tag_keep_missing_override,
                                  const std::optional<int32_t>& min_mapq_override,
                                  const std::optional<secondary::HaplotagSource>& haptag_source,
                                  const std::optional<std::filesystem::path>& phasing_bin_fn,
                                  const secondary::KadayashiOptions& kadayashi_opt,
                                  bool legacy_feature_gen);

/**
 * \brief Reads BAM windows, haplotags reads, calls simple variants and builds inference samples.
 *          Additionally, accumulates simple variants per chromosome.
 * \param input_queue Queue of BAM windows to encode.
 * \param output_queue Queue that receives encoded inference samples.
 * \param chrom_reduce_data Per-chromosome reduction state updated as windows are processed.
 * \param resources Shared variant-calling resources, including encoders and models.
 * \param stats Shared progress statistics updated as BAM windows complete.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param bam_regions Input BAM windows grouped by chromosome.
 * \param draft_lens Draft/reference names and lengths in seq_id order.
 * \param candidate_source Source of candidate variant positions.
 * \param candidate_trees_from_file Optional candidate-site interval trees loaded from file input.
 * \param num_threads Number of worker threads available to the producer stage.
 * \param window_len Target inference window length.
 * \param window_overlap Overlap between adjacent inference windows.
 * \param variant_flanking_bases Number of flanking bases to preserve around candidate positions.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 * \param ploidy Ploidy used for simple variant conversion.
 * \param pass_min_qual Minimum QUAL required for a simple variant to be treated as PASS.
 * \param tiled_regions Whether candidate-aware tiled splitting is enabled. Otherwise, samples are constructed around candidate variants if used.
 * \param tiled_ext_flanks Whether tiled windows should be extended with extra flanks.
 * \param tiled_ext_major Major-coordinate extension for tiled windows.
 * \param tiled_ext_min_cov Minimum coverage threshold for tiled window extension.
 * \param tiled_ext_cov_fract Coverage-fraction threshold for tiled window extension.
 * \param min_depth Minimum depth required for inference columns to be retained.
 */
void worker_sample_producer(
        utils::AsyncQueue<secondary::Window>& input_queue,
        utils::AsyncQueue<InferenceData>& output_queue,
        std::vector<ChromosomeReduceData>& chrom_reduce_data,
        VariantResources& resources,
        secondary::Stats& stats,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        const std::vector<std::vector<secondary::Window>>& bam_regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<std::string>& draft_seqs,
        const secondary::VariantCandidateSource candidate_source,
        const std::optional<secondary::IntervalTreesInt64Map>& candidate_trees_from_file,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t variant_flanking_bases,
        const bool continue_on_exception,
        const int32_t ploidy,
        const float pass_min_qual,
        const bool tiled_regions,
        const bool tiled_ext_flanks,
        const int64_t tiled_ext_major,
        const int64_t tiled_ext_min_cov,
        const float tiled_ext_cov_fract,
        const int32_t min_depth);

/**
 * \brief Collates encoded samples into inference batches that fit within the available memory
 *        budget.
 * \param input_queue Queue of encoded samples ready for batching.
 * \param output_queue Queue that receives batches for model inference.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param model Model instance used to estimate batch memory requirements.
 * \param window_len Target inference window length.
 * \param batch_size Requested batch size.
 * \param max_available_mem Maximum memory budget available for a batch.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 */
void worker_batch_producer(utils::AsyncQueue<InferenceData>& input_queue,
                           utils::AsyncQueue<InferenceData>& output_queue,
                           std::atomic<bool>& worker_terminate,
                           secondary::WorkerReturnStatus& ret_status,
                           const dorado::secondary::ModelTorchBase& model,
                           const int32_t window_len,
                           const int32_t batch_size,
                           const double max_available_mem,
                           const bool continue_on_exception);

/**
 * \brief Runs model inference for each batch and forwards raw decode data to the next stage.
 * \param batch_queue Queue of collated inference batches.
 * \param decode_queue Queue that receives raw model outputs for later decoding.
 * \param models Model instances used for inference.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param streams Optional device streams used to overlap inference work.
 * \param encoders Encoders associated with the models, used for shape and metadata handling.
 * \param draft_lens Draft/reference names and lengths in seq_id order.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 */
void worker_infer_samples_in_parallel(
        utils::AsyncQueue<InferenceData>& batch_queue,
        utils::AsyncQueue<DecodeData>& decode_queue,
        std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        const std::vector<c10::optional<c10::Stream>>& streams,
        const std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool continue_on_exception);

/**
 * \brief Splits batched decode output into per-sample variant-calling inputs.
 * \param input_queue Queue of decoded batch outputs.
 * \param out_queue Queue that receives per-sample variant-calling inputs.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param num_threads Number of worker threads associated with the upstream inference stage.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 */
void worker_separate_decode_data(utils::AsyncQueue<DecodeData>& input_queue,
                                 utils::AsyncQueue<secondary::VariantCallingSample>& out_queue,
                                 std::atomic<bool>& worker_terminate,
                                 secondary::WorkerReturnStatus& ret_status,
                                 const int32_t num_threads,
                                 const bool continue_on_exception);

/**
 * \brief Collects variant-calling samples by chromosome, decodes inference variants, merges them
 *        with simple variants, and queues completed chromosomes for writing.
 * \param input_queue Queue of per-sample variant-calling inputs.
 * \param out_queue Queue that receives chromosome IDs ready for output writing.
 * \param chrom_reduce_data Per-chromosome reduction state updated by this stage.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param stats Shared progress statistics updated from incoming inferred samples and any
 *              per-sequence remainder finalized once reduction completes.
 * \param fastx_readers Indexed FASTX readers, one per worker thread.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 * \param num_threads Number of worker threads assigned to this reduction stage.
 * \param draft_lens Draft/reference names and lengths in seq_id order.
 * \param decoder Decoder used to convert model outputs into consensus symbols.
 * \param hemizygous_regions Per-chromosome vector of regions where we should output a haploid call.
 * \param pass_min_qual Minimum QUAL required to mark decoded variants as PASS.
 * \param ambig_ref Whether ambiguous reference bases are allowed during variant decoding.
 * \param gvcf Whether non-variant positions should be emitted as gVCF records.
 * \param ploidy Ploidy used for gVCF reference genotypes.
 * \param flank_trim Number of bases to trim from each processed interval before merge filtering.
 * \param variant_candidate_source Source of candidate variant positions.
 */
void worker_variant_calling_reduce(
        utils::AsyncQueue<secondary::VariantCallingSample>& input_queue,
        utils::AsyncQueue<int64_t>& out_queue,  // ID of the chromosome that is ready for writing.
        std::vector<ChromosomeReduceData>& chrom_reduce_data,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        secondary::Stats& stats,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& fastx_readers,
        const bool continue_on_exception,
        const int32_t num_threads,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        const std::vector<std::vector<secondary::Region>>& hemizygous_regions,
        const float pass_min_qual,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t ploidy,
        const int32_t flank_trim,
        const secondary::VariantCandidateSource variant_candidate_source);

/**
 * \brief Writes completed chromosome results to the merged VCF, optional auxiliary VCFs, and the
 *        processed-regions BED output.
 * \param input_queue Queue of chromosome IDs whose results are ready to be written.
 * \param chrom_reduce_data Per-chromosome reduction state containing buffered output records.
 * \param worker_terminate Shared termination flag for the async pipeline.
 * \param ret_status Worker status populated when the stage fails.
 * \param vcf_writer Writer for the merged output VCF/gVCF.
 * \param ofs_regions Stream used to write processed regions in BED format.
 * \param vcf_writer_simple Optional writer for simple/Kadayashi-only variants.
 * \param vcf_writer_inference Optional writer for inference-only variants.
 * \param continue_on_exception Whether worker exceptions should be logged and skipped.
 */
void worker_variant_writer(
        utils::AsyncQueue<int64_t>& input_queue,  // ID of the chromosome that is ready for writing.
        std::vector<ChromosomeReduceData>&
                chrom_reduce_data,  // Nonconst so that data can be freed.
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        secondary::VCFWriter& vcf_writer,
        std::ofstream& ofs_regions,
        std::optional<secondary::VCFWriter>& vcf_writer_simple,
        std::optional<secondary::VCFWriter>& vcf_writer_inference,
        const bool continue_on_exception);

/**
 * \brief Converts Kadayashi variants into Dorado's normalized secondary::Variant representation.
 * \param kadayashi_variants Input variants emitted by Kadayashi.
 * \param seq_id Sequence ID associated with the variants.
 * \param ploidy Ploidy used to normalize genotype information.
 * \param pass_min_qual Minimum QUAL required to mark a converted variant as PASS.
 * \return Converted variants.
 */
std::vector<secondary::Variant> convert_variants(
        const std::vector<kadayashi::variant_dorado_style_t>& kadayashi_variants,
        int32_t seq_id,
        int32_t ploidy,
        float pass_min_qual);

/**
 * \brief Selects variants according to overlaps with hemizygous_regions and passes them to
          secondary::collapse_to_haploid.
 * \param variants Input variants for a single contig.
 * \param hemizygous_regions Regions in which to output hemizygous calls for a single contig, possibly empty.
 * \return Filtered and collapsed variants.
 */
std::vector<secondary::Variant> filter_hemizygous_variants(
        const std::vector<secondary::Variant>& variants,
        const std::vector<secondary::Region>& hemizygous_regions);

}  // namespace dorado::smallvar
