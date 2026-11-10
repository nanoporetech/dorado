#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "dorado_version.h"
#include "hts_utils/FastxRandomReader.h"
#include "hts_utils/fai_utils.h"
#include "model_downloader/model_downloader.h"
#include "model_resolver/ModelResolver.h"
#include "models/models.h"
#include "secondary/architectures/model_config.h"
#include "secondary/common/bam_info.h"
#include "secondary/common/region.h"
#include "secondary/common/resolve_model.h"
#include "secondary/common/stats.h"
#include "secondary/common/variant.h"
#include "secondary/common/vcf_writer.h"
#include "secondary/consensus/window_utils.h"
#include "secondary/features/feature_columns.h"
#include "secondary/features/haplotag_source.h"
#include "secondary/features/variant_candidate_source.h"
#include "smallvar_progress_tracker.h"
#include "torch_utils/torch_utils.h"
#include "utils/AsyncQueue.h"
#include "utils/jthread.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"
#include "utils/thread_utils.h"
#include "variant/variant_impl.h"

#include <IntervalTree.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace dorado {

namespace {

enum class VariantCallingFormatEnum {
    VCF,
    GVCF,
};

/// \brief All options for this tool.
struct Options {
    // Positional parameters.
    std::filesystem::path in_aln_bam_fn;
    std::filesystem::path in_ref_fastx_fn;

    // Optional parameters.
    std::filesystem::path output_dir;
    VariantCallingFormatEnum out_format = VariantCallingFormatEnum::VCF;
    std::string model_str;
    std::optional<std::filesystem::path> models_directory;
    int32_t verbosity = 0;
    int32_t threads = 0;
    int32_t infer_threads = 1;
    std::string device_str;
    int32_t batch_size = 10;
    std::optional<int32_t> window_len{};
    std::optional<int32_t> window_overlap{};
    std::optional<int32_t> variant_flanking_bases{};
    int32_t bam_chunk = 1'000'000;
    std::optional<std::string> regions_str;
    std::vector<secondary::Region> regions;
    std::optional<std::string> hemizygous_regions_str;
    std::vector<secondary::Region> hemizygous_regions;
    bool full_precision = false;
    bool load_scripted_model = false;
    std::optional<int32_t> bam_region_queue_size;
    std::optional<int32_t> sample_queue_size;
    std::optional<int32_t> batch_queue_size;
    std::optional<int32_t> decode_queue_size;
    std::optional<int32_t> vc_data_queue_size;
    std::optional<int32_t> vc_writer_queue_size;
    std::optional<int32_t> min_mapq;
    std::string read_group;
    bool ignore_read_groups = false;
    std::string tag_name;
    int32_t tag_value = 0;
    std::optional<bool> tag_keep_missing;  // Optionally overrides the model config if specified.
    int32_t min_depth = 0;
    bool any_bam = false;
    bool ambig_ref = false;
    float pass_min_qual = 0.0f;
    float pass_min_gq = 0.0f;
    float pass_min_aq = 0.0f;
    float pass_min_vq = 4.5f;
    std::optional<std::string> sample_name{};

    secondary::HaplotagSource haplotag_source = secondary::HaplotagSource::COMPUTE;
    std::optional<std::filesystem::path> phasing_bin_path;
    bool hp_tag_from_bam = false;
    bool unphased = false;

    bool continue_on_error = false;

    double min_snp_accuracy = 0.0;
    bool tiled_regions = true;      // Candidate region selection using a tiled approach.
    bool tiled_ext_flanks = true;   // Select neighboring windows if there are deletions in flanks.
    int32_t tiled_ext_major = 10;   // Number of flanking major positions to check for the trigger.
    int32_t tiled_ext_min_cov = 3;  // Minimum deletion coverage to trigger the extension.
    float tiled_ext_cov_fract = 0.25f;  // Fraction of deletion coverage to trigger the heuristic.

    // Candidate region filtering based on candidate variants.
    // Optionally force candidate region selection on. Otherwise the model config default is used.
    std::optional<bool> candidate_filtering;
    std::optional<std::filesystem::path> candidate_variants_path;
    int32_t flank_trim_len = 5;

    secondary::KadayashiOptions kadayashi_opt;
    bool dump_variants = false;
    bool legacy_feature_gen = false;
};

struct QueueSizes {
    size_t bam_region{};
    size_t sample{};
    size_t batch{};
    size_t decode{};
    size_t vc_data{};
    size_t vc_writer{};
};

QueueSizes resolve_queue_sizes(const Options& opt, const size_t num_inference_workers) {
    constexpr size_t QUEUED_ITEMS_PER_WORKER = 2;
    constexpr size_t BATCHES_PER_INFERENCE_WORKER = 5;
    constexpr size_t DECODE_BATCHES_PER_WORKER = 2;
    constexpr size_t WRITER_QUEUE_SIZE = 2;

    const size_t num_feature_workers = static_cast<size_t>(opt.threads);
    const size_t num_reduce_workers = static_cast<size_t>(opt.threads);

    const auto resolve = [](const std::optional<int32_t>& stage_size, const size_t default_size) {
        return stage_size ? static_cast<size_t>(*stage_size) : default_size;
    };

    return {
            .bam_region = resolve(opt.bam_region_queue_size,
                                  QUEUED_ITEMS_PER_WORKER * num_feature_workers),
            .sample = resolve(opt.sample_queue_size, QUEUED_ITEMS_PER_WORKER * num_feature_workers),
            .batch = resolve(opt.batch_queue_size,
                             BATCHES_PER_INFERENCE_WORKER * num_inference_workers),
            .decode = resolve(opt.decode_queue_size,
                              DECODE_BATCHES_PER_WORKER * num_inference_workers),
            .vc_data =
                    resolve(opt.vc_data_queue_size, QUEUED_ITEMS_PER_WORKER * num_reduce_workers),
            .vc_writer = resolve(opt.vc_writer_queue_size, WRITER_QUEUE_SIZE),
    };
}

bool parse_bool_arg(const std::string& raw_value) {
    std::string value = raw_value;
    std::transform(std::begin(raw_value), std::end(raw_value), std::begin(value),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if ((value == "true") || (value == "1") || (value == "yes") || (value == "on")) {
        return true;
    }
    if ((value == "false") || (value == "0") || (value == "no") || (value == "off")) {
        return false;
    }

    throw std::runtime_error{"Invalid value for --candidate-filtering: '" + raw_value +
                             "'. Expected true/false, 1/0, yes/no, or on/off."};
}

/// \brief Define the CLI options.
void add_arguments(argparse::ArgumentParser& parser, int& verbosity) {
    parser.add_description("Diploid variant calling tool");

    {
        // Positional arguments group
        parser.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser.add_argument("in_ref_fastx").help("Reference sequences");
    }
    {
        // Default "Optional arguments" group
        parser.add_argument("-t", "--threads")
                .help("Number of threads for processing (0=unlimited).")
                .default_value(0)
                .scan<'i', int>();

        parser.add_argument("--infer-threads")
                .help("Number of threads for inference")
#if DORADO_CUDA_BUILD
                .default_value(2)
#else
                .default_value(1)
#endif
                .scan<'i', int>();

        parser.add_argument("-x", "--device")
                .help(std::string{"Specify CPU or GPU device: 'auto', 'cpu', 'cuda:all' or "
                                  "'cuda:<device_id>[,<device_id>...]'. The 'cuda' option is only "
                                  "available on CUDA-enabled systems. Specifying 'auto' will "
                                  "choose either 'cpu' or 'cuda:all' depending on the presence of "
                                  "a GPU device."})
                .default_value(std::string{"auto"});

        parser.add_argument("-v", "--verbose")
                .flag()
                .action([&](const auto&) { ++verbosity; })
                .append();
    }
    {
        parser.add_group("Input/output options");
        parser.add_argument("-o", "--output-dir")
                .help("If specified, output files will be written to the given folder. Otherwise, "
                      "output is to stdout.")
                .default_value("");
        parser.add_argument("--models-directory")
                .help("Optional directory to search for existing models or download new models "
                      "into.");
        parser.add_argument("--gvcf").help("Output a gVCF instead of a VCF.").flag();
        parser.add_argument("--ambig-ref")
                .help("Decode variants at ambiguous reference positions.")
                .flag();
    }
    {
        parser.add_group("Advanced options");
        parser.add_argument("-b", "--batchsize")
                .help("Batch size for inference. Default: 0 for auto batch size detection.")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("--bam-chunk")
                .help("Size of reference chunks to parse from the input BAM at a time.")
                .default_value(1000000)
                .scan<'i', int>();
        parser.add_argument("--regions")
                .help("Process only these regions of the input. Can be either a path to a BED file "
                      "or a list of comma-separated Htslib-formatted regions (start is 1-based, "
                      "end "
                      "is inclusive).");
        parser.add_argument("--hemizygous-regions")
                .help("Regions in which haploid variant calls are expected. Can be either a path "
                      "to a BED file "
                      "or a list of comma-separated Htslib-formatted regions (start is 1-based, "
                      "end "
                      "is inclusive).");
        parser.add_argument("--RG").help("Read group to select.").default_value("");
        parser.add_argument("--ignore-read-groups").help("Ignore read groups in bam file.").flag();
        parser.add_argument("--tag-name")
                .help("Two-letter BAM tag name for filtering the alignments during feature "
                      "generation")
                .default_value("");
        parser.add_argument("--tag-value")
                .help("Value of the tag for filtering the alignments during feature generation")
                .default_value(0)
                .scan<'i', int>();

        parser.add_argument("--tag-keep-missing")
                .help("Keep alignments when tag is missing. If specified, overrides "
                      "the same option in the model config.")
                .flag();
        parser.add_argument("--min-mapq")
                .help("Minimum mapping quality of the input alignments. If specified, overrides "
                      "the same option in the model config.")
                .scan<'i', int>();
        parser.add_argument("--min-depth")
                .help("Sites with depth lower than this value will not be processed.")
                .default_value(1)
                .scan<'i', int>();
        parser.add_argument("--pass-qual-filter")
                .help("Set QUAL filter for PASS variants.")
                .default_value(0.0f)
                .scan<'g', float>();
        parser.add_argument("--pass-gq-filter")
                .help("Set GQ filter for PASS variants.")
                .default_value(0.0f)
                .scan<'g', float>();
        parser.add_argument("--pass-aq-filter")
                .help("Set AQ filter for PASS variants.")
                .default_value(0.0f)
                .scan<'g', float>();
        parser.add_argument("--pass-vq-filter")
                .help("Set VQ filter for PASS variants.")
                .default_value(4.5f)
                .scan<'g', float>();
        parser.add_argument("--sample-name").help("Set sample name in VCF.");
        parser.add_argument("--model-override")
                .help("Path to a specific model folder. Overrides auto model resolution and all "
                      "compatibility checks. This may produce inferior results.")
                .default_value("");
    }
    {
        parser.add_group("Phasing options");
        parser.add_argument("--hp-tag")
                .help("Use the HP tag from the input BAM file to load phasing information, instead "
                      "of computing it.")
                .flag();
        parser.add_argument("--phasing-bin")
                .hidden()
                .help("The .bin file containing phasing information for reads in the given BAM.");
        parser.add_argument("--unphased").help("Deactivate phasing.").flag().default_value(false);
    }

    // Hidden advanced arguments.
    {
        parser.add_argument("--window-len")
                .hidden()
                .help("Overrides the model-defined window (chunk) size for inference.")
                .scan<'i', int>();
        parser.add_argument("--window-overlap")
                .hidden()
                .help("Overrides the model-defined window (chunk) overlap length for inference.")
                .scan<'i', int>();
        parser.add_argument("--full-precision")
                .hidden()
                .help("Always use full precision for inference.")
                .flag();
        parser.add_argument("--bam-region-queue-size")
                .hidden()
                .help("Override the BAM-region queue size.")
                .scan<'i', int>();
        parser.add_argument("--sample-queue-size")
                .hidden()
                .help("Override the encoded-sample queue size.")
                .scan<'i', int>();
        parser.add_argument("--batch-queue-size")
                .hidden()
                .help("Override the inference-batch queue size.")
                .scan<'i', int>();
        parser.add_argument("--decode-queue-size")
                .hidden()
                .help("Override the inference-output queue size.")
                .scan<'i', int>();
        parser.add_argument("--vc-data-queue-size")
                .hidden()
                .help("Override the variant-calling sample queue size.")
                .scan<'i', int>();
        parser.add_argument("--vc-writer-queue-size")
                .hidden()
                .help("Override the variant-writer notification queue size.")
                .scan<'i', int>();
        parser.add_argument("--scripted")
                .hidden()
                .help("Load the scripted Torch model instead of building one internally.")
                .flag();
        parser.add_argument("--any-bam")
                .hidden()
                .help("Allow any BAM as input, not just Dorado aligned.")
                .flag();
        parser.add_argument("--continue-on-error")
                .hidden()
                .help("Continue the process even if an exception is thrown. This "
                      "may leave some regions unprocessed.")
                .flag();
        parser.add_argument("--min-snp-acc")
                .hidden()
                .help("Filter alignments with SNP accuracy below this threshold in range [0.0, "
                      "1.0].")
                .default_value(0.0f)
                .scan<'g', float>();
        // Candidate region selection options.
        parser.add_argument("--candidate-filtering")
                .help("Overrides the model-defined candidate region filtering feature. Set to true "
                      "to limit inference to regions containing uncertain variant candidates, or "
                      "false to run dense inference.")
                .metavar("BOOL")
                .action(parse_bool_arg);
        parser.add_argument("--candidates")
                .hidden()
                .help("Path to a tab-separated file containing coordinates of variant candidate "
                      "sites to process.");
        parser.add_argument("--variant-flanking-bases")
                .hidden()
                .help("Minimum number of flanking bases in samples around candidate variants.")
                .scan<'i', int>();
        parser.add_argument("--candidate-centered-regions")
                .hidden()
                .help("Construct regions centered around candidate variants instead of using the "
                      "default tiled approach. Uses --variant-flanking-bases instead of "
                      "--window-overlap.")
                .flag()
                .default_value(false);
        parser.add_argument("--no-tiled-ext-flanks")
                .hidden()
                .help("Disable the heuristic that additionally processes neighboring windows if "
                      "selected tiled windows have deletions in the flanks.")
                .flag()
                .default_value(false);
        parser.add_argument("--tiled-ext-major")
                .hidden()
                .help("Number of major bases to check in the flanks to trigger the extension "
                      "heuristic.")
                .default_value(10)
                .scan<'i', int>();
        parser.add_argument("--tiled-ext-min-cov")
                .hidden()
                .help("Minimum absolute deletion coverage to trigger the extension heuristic.")
                .default_value(3)
                .scan<'i', int>();
        parser.add_argument("--tiled-ext-cov-fract")
                .hidden()
                .help("Minimum coverage fraction of deletions in the major columns to trigger the "
                      "extension heuristic.")
                .default_value(0.25f)
                .scan<'g', float>();
        parser.add_argument("--flank-trim-len")
                .hidden()
                .help("Removes inference variants within this many bases from every edge of every "
                      "processed region. Only applied when simple variants are computed and merged "
                      "internally (i.e. no `--candidates` path is specified).")
                .default_value(3)
                .scan<'i', int>();

        // Kadayashi options.
        parser.add_argument("--kada-disable-interval-exp")
                .hidden()
                .help("Disable interval expansion for Kadayashi haplotagging/variant calling.")
                .flag();
        parser.add_argument("--kada-min-base-qual")
                .hidden()
                .help("Minimum base quality for Kadayashi haplotagging/variant calling.")
                .default_value(5)
                .scan<'i', int>();
        parser.add_argument("--kada-min-varcall-cov")
                .hidden()
                .help("Minimum base variant calling coverage for Kadayashi.")
                .default_value(5)
                .scan<'i', int>();
        parser.add_argument("--kada-min-varcall-fract")
                .hidden()
                .help("Minimum coverage fraction for Kadayashi haplotagging/variant calling.")
                .default_value(0.2f)
                .scan<'g', float>();
        parser.add_argument("--kada-max-clipping")
                .hidden()
                .help("Maximum alignment clipping for Kadayashi haplotagging/variant calling.")
                .default_value(10000000)
                .scan<'i', int>();
        parser.add_argument("--kada-min-strand-cov")
                .hidden()
                .help("Minimum strand coverage Kadayashi haplotagging/variant calling.")
                .default_value(3)
                .scan<'i', int>();
        parser.add_argument("--kada-min-strand-cov-fract")
                .hidden()
                .help("Minimum strand coverage fraction for Kadayashi haplotagging/variant "
                      "calling.")
                .default_value(0.03f)
                .scan<'g', float>();
        parser.add_argument("--kada-max-gapcomp-seq-div")
                .hidden()
                .help("Maximum gapcompressed seqdiv for Kadayashi haplotagging/variant calling.")
                .default_value(0.1f)
                .scan<'g', float>();
        parser.add_argument("--dump-variants")
                .hidden()
                .help("Write individual Kadayashi and inference variants if the output is to a "
                      "folder.")
                .flag();
        parser.add_argument("--legacy-feature-gen")
                .hidden()
                .help("Use legacy read alignment feature generation.")
                .flag();
    }
}

int parse_args(int argc, char** argv, argparse::ArgumentParser& parser) {
    try {
        cli::parse(parser, argc, argv);

    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        DORADO_LOG(error, "{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/// \brief This function simply fills out the Options struct with the parsed CLI args.
Options set_options(const argparse::ArgumentParser& parser, const int verbosity) {
    Options opt;

    opt.in_aln_bam_fn = parser.get<std::string>("in_aln_bam");
    opt.in_ref_fastx_fn = parser.get<std::string>("in_ref_fastx");

    opt.output_dir = parser.get<std::string>("output-dir");
    opt.models_directory = model_resolution::get_models_directory(
            cli::get_optional_argument<std::string>("--models-directory", parser));
    opt.model_str = parser.get<std::string>("model-override");
    opt.out_format = parser.get<bool>("gvcf") ? VariantCallingFormatEnum::GVCF
                                              : VariantCallingFormatEnum::VCF;
    opt.threads = parser.get<int>("threads");
    opt.threads = (opt.threads == 0) ? std::thread::hardware_concurrency() : (opt.threads);
    opt.infer_threads = parser.get<int>("infer-threads");

    opt.device_str = parser.get<std::string>("device");
    if (opt.device_str == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        opt.device_str = "cpu";
#else
        opt.device_str = utils::get_auto_detected_device();
#endif
    }

    opt.batch_size = parser.get<int>("batchsize");

    opt.bam_chunk = parser.get<int>("bam-chunk");
    opt.verbosity = verbosity;
    opt.regions_str = parser.present<std::string>("regions");
    if (opt.regions_str) {
        opt.regions = secondary::parse_regions(*opt.regions_str);
    }
    opt.hemizygous_regions_str = parser.present<std::string>("hemizygous-regions");
    if (opt.hemizygous_regions_str) {
        opt.hemizygous_regions = secondary::parse_regions(*opt.hemizygous_regions_str);
    }
    opt.min_depth = parser.get<int>("min-depth");

    opt.window_len = parser.present<int32_t>("window-len");
    opt.window_overlap = parser.present<int32_t>("window-overlap");
    opt.variant_flanking_bases = parser.present<int32_t>("variant-flanking-bases");
    opt.full_precision = parser.get<bool>("full-precision");
    opt.load_scripted_model = parser.get<bool>("scripted");
    opt.bam_region_queue_size = parser.present<int32_t>("bam-region-queue-size");
    opt.sample_queue_size = parser.present<int32_t>("sample-queue-size");
    opt.batch_queue_size = parser.present<int32_t>("batch-queue-size");
    opt.decode_queue_size = parser.present<int32_t>("decode-queue-size");
    opt.vc_data_queue_size = parser.present<int32_t>("vc-data-queue-size");
    opt.vc_writer_queue_size = parser.present<int32_t>("vc-writer-queue-size");
    opt.any_bam = parser.get<bool>("any-bam");
    opt.continue_on_error = parser.get<bool>("continue-on-error");
    opt.read_group = (parser.is_used("--RG")) ? parser.get<std::string>("RG") : "";
    opt.ignore_read_groups = parser.get<bool>("ignore-read-groups");
    opt.tag_name = parser.get<std::string>("tag-name");
    opt.tag_value = parser.get<int>("tag-value");
    // The `"--tag-keep-missing` is a special case because it's a flag, and we cannot use `present`.
    opt.tag_keep_missing = (parser.is_used("--tag-keep-missing"))
                                   ? std::optional<bool>{parser.get<bool>("tag-keep-missing")}
                                   : std::nullopt;
    opt.min_mapq = parser.present<int32_t>("min-mapq");
    opt.ambig_ref = parser.get<bool>("ambig-ref");
    opt.sample_name = cli::get_optional_argument<std::string>("--sample-name", parser);
    opt.pass_min_qual = parser.get<float>("pass-qual-filter");
    opt.pass_min_gq = parser.get<float>("pass-gq-filter");
    opt.pass_min_aq = parser.get<float>("pass-aq-filter");
    opt.pass_min_vq = parser.get<float>("pass-vq-filter");

    opt.candidate_variants_path = parser.present<std::string>("candidates");

    // Override the candidate filtering option.
    opt.candidate_filtering = parser.present<bool>("candidate-filtering");

    opt.phasing_bin_path = parser.present<std::string>("phasing-bin");
    opt.hp_tag_from_bam = parser.get<bool>("hp-tag");
    opt.unphased = parser.get<bool>("unphased");
    opt.haplotag_source = (opt.phasing_bin_path)  ? secondary::HaplotagSource::BIN_FILE
                          : (opt.hp_tag_from_bam) ? secondary::HaplotagSource::BAM_HAP_TAG
                          : (opt.unphased)        ? secondary::HaplotagSource::UNPHASED
                                                  : secondary::HaplotagSource::COMPUTE;

    opt.tiled_regions = !parser.get<bool>("candidate-centered-regions");
    opt.tiled_ext_flanks = !parser.get<bool>("no-tiled-ext-flanks");
    opt.tiled_ext_major = parser.get<int>("tiled-ext-major");
    opt.tiled_ext_min_cov = parser.get<int>("tiled-ext-min-cov");
    opt.tiled_ext_cov_fract = parser.get<float>("tiled-ext-cov-fract");

    opt.flank_trim_len = parser.get<int>("flank-trim-len");

    opt.min_snp_accuracy = parser.get<float>("min-snp-acc");

    opt.kadayashi_opt.disable_interval_expansion = parser.get<bool>("kada-disable-interval-exp");
    opt.kadayashi_opt.min_base_quality = parser.get<int>("kada-min-base-qual");
    opt.kadayashi_opt.min_varcall_coverage = parser.get<int>("kada-min-varcall-cov");
    opt.kadayashi_opt.min_varcall_fraction = parser.get<float>("kada-min-varcall-fract");
    opt.kadayashi_opt.max_clipping = parser.get<int>("kada-max-clipping");
    opt.kadayashi_opt.min_strand_cov = parser.get<int>("kada-min-strand-cov");
    opt.kadayashi_opt.min_strand_cov_frac = parser.get<float>("kada-min-strand-cov-fract");
    opt.kadayashi_opt.max_gapcompressed_seqdiv = parser.get<float>("kada-max-gapcomp-seq-div");
    opt.kadayashi_opt.ambig_ref = opt.ambig_ref;

    opt.dump_variants = parser.get<bool>("dump-variants");
    opt.legacy_feature_gen = parser.get<bool>("legacy-feature-gen");

    return opt;
}

void validate_options(const Options& opt) {
    // Parameter validation.
    if (!cli::validate_device_string(opt.device_str)) {
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_aln_bam_fn) ||
        std::filesystem::is_empty(opt.in_aln_bam_fn)) {
        DORADO_LOG(error, "Input file '{}' does not exist or is empty.",
                   opt.in_aln_bam_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_ref_fastx_fn) ||
        std::filesystem::is_empty(opt.in_ref_fastx_fn)) {
        DORADO_LOG(error, "Input file '{}' does not exist or is empty.",
                   opt.in_ref_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (opt.batch_size < 0) {
        DORADO_LOG(error, "Batch size should be >= 0. Given: {}.", opt.batch_size);
        std::exit(EXIT_FAILURE);
    }
    if (opt.bam_chunk <= 0) {
        DORADO_LOG(error, "BAM chunk size should be > 0. Given: {}.", opt.bam_chunk);
        std::exit(EXIT_FAILURE);
    }

    const auto validate_queue_size = [](const std::string_view option_name,
                                        const std::optional<int32_t>& queue_size) {
        if (queue_size && (*queue_size <= 0)) {
            DORADO_LOG(error, "{} needs to be > 0, given: {}.", option_name, *queue_size);
            std::exit(EXIT_FAILURE);
        }
    };
    validate_queue_size("--bam-region-queue-size", opt.bam_region_queue_size);
    validate_queue_size("--sample-queue-size", opt.sample_queue_size);
    validate_queue_size("--batch-queue-size", opt.batch_queue_size);
    validate_queue_size("--decode-queue-size", opt.decode_queue_size);
    validate_queue_size("--vc-data-queue-size", opt.vc_data_queue_size);
    validate_queue_size("--vc-writer-queue-size", opt.vc_writer_queue_size);

    if ((std::size(opt.tag_name) == 1) || (std::size(opt.tag_name) > 2)) {
        DORADO_LOG(error,
                   "The --tag-name is specified, but it needs to contain exactly two characters. "
                   "Given: '{}'.",
                   opt.tag_name);
        std::exit(EXIT_FAILURE);
    }

    if (opt.regions_str && std::empty(opt.regions)) {
        DORADO_LOG(error, "Option --regions is specified, but an empty set of regions is given!");
        std::exit(EXIT_FAILURE);
    }

    if (opt.hemizygous_regions_str && std::empty(opt.hemizygous_regions)) {
        DORADO_LOG(
                error,
                "Option --hemizygous-regions is specified, but an empty set of regions is given!");
        std::exit(EXIT_FAILURE);
    }

    if (opt.phasing_bin_path && !std::filesystem::exists(*opt.phasing_bin_path)) {
        DORADO_LOG(error, "Phasing bin path file '{}' does not exist.",
                   opt.phasing_bin_path->string());
        std::exit(EXIT_FAILURE);
    }

    // If multiple opposing haplotagging options are selected, report an error.
    if (((opt.phasing_bin_path != std::nullopt) + opt.hp_tag_from_bam + opt.unphased) > 1) {
        DORADO_LOG(error,
                   "Selected multiple haplotagging options (phasing-bin, hp-tag or unphased). Only "
                   "one is allowed.");
        std::exit(EXIT_FAILURE);
    }

    if (opt.candidate_variants_path && !std::filesystem::exists(*opt.candidate_variants_path)) {
        DORADO_LOG(error, "Candidate site file '{}' does not exist.",
                   opt.candidate_variants_path->string());
        std::exit(EXIT_FAILURE);
    }

    if (opt.dump_variants && std::empty(opt.output_dir)) {
        DORADO_LOG(
                error,
                "The --dump-variants option only works when the output is to a directory, but the "
                "output directory is not specified.");
        std::exit(EXIT_FAILURE);
    }

    if ((opt.min_snp_accuracy < 0.0) || (opt.min_snp_accuracy > 1.0)) {
        DORADO_LOG(error,
                   "The --min-snp-acc value needs to be in range [0.0, 1.0]. Specified: '{}'.",
                   opt.min_snp_accuracy);
    }

    if (opt.candidate_variants_path && !std::filesystem::exists(*opt.candidate_variants_path)) {
        DORADO_LOG(error, "Candidate site file '{}' does not exist.",
                   opt.candidate_variants_path->string());
        std::exit(EXIT_FAILURE);
    }
}

int32_t count_model_hits(std::span<const dorado::models::ModelInfo> model_list,
                         const std::string& model_name) {
    return static_cast<int32_t>(std::count_if(
            std::begin(model_list), std::end(model_list),
            [&model_name](const models::ModelInfo& info) { return info.name == model_name; }));
}

std::string determine_model_name(const std::string& basecaller_model, const bool use_dwells) {
    auto find_model = [](const std::string& model_prefix) {
        std::string ret;
        for (const auto& info : models::variant_models()) {
            // Variant models can have multiple versions for one basecaller. The list is ordered so
            // that the last matching entry is the latest compatible model.
            if (utils::starts_with(info.name, model_prefix)) {
                ret = info.name;
            }
        }
        return ret;
    };

    const std::string smallvar_model_prefix =
            basecaller_model + (use_dwells ? "_smallvar_mv@" : "_smallvar@");

    const std::string ret = find_model(smallvar_model_prefix);

    if (std::empty(ret)) {
        throw std::runtime_error{
                "Could not find any smallvar model compatible with the basecaller model '" +
                basecaller_model + "'."};
    }

    return ret;
}

std::filesystem::path resolve_model(const secondary::BamInfo& bam_info,
                                    const std::optional<std::filesystem::path>& models_directory) {
    DORADO_LOG(info, "Auto resolving the model.");

    if (bam_info.has_dwells) {
        DORADO_LOG(info, "Input data contains move tables.");
    } else {
        DORADO_LOG(info, "Input data does not contain move tables.");
    }

    throw_unless_single_model_present(bam_info);

    check_stereo_models(bam_info, true);

    // Example: dna_r10.4.1_e8.2_400bps_hac@v5.2.0
    const std::string& basecaller_model = *std::begin(bam_info.basecaller_models);

    // Example: dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0
    const std::string model_name = determine_model_name(basecaller_model, false);

    // Sanity check that the model name exists in the smallvar models.
    if (count_model_hits(models::variant_models(), model_name) == 0) {
        throw std::runtime_error{"Resolved model '" + model_name + "' not found!"};
    }

    DORADO_LOG(debug, "Resolved model from input data: '{}'", model_name);

    model_downloader::ModelDownloader downloader(models_directory, false);
    const std::filesystem::path model_dir = downloader.get(model_name, "smallvar");

    return model_dir;
}

std::filesystem::path resolve_model_manually(
        const secondary::BamInfo& bam_info,
        const std::optional<std::filesystem::path>& models_directory,
        const std::string& model_str,
        const bool any_model) {
    DORADO_LOG(info, "User-specificed model.");

    if (bam_info.has_dwells) {
        DORADO_LOG(info, "Input data contains move tables.");
    } else {
        DORADO_LOG(info, "Input data does not contain move tables.");
    }

    // Fail only if not explicitly permitting any model.
    if (!any_model) {
        throw_unless_single_model_present(bam_info);
    }

    check_stereo_models(bam_info, !any_model);

    std::filesystem::path model_dir;

    if (!std::empty(model_str)) {
        DORADO_LOG(
                warn,
                "Skipping basecaller compatibility checks for user-specified model override. The "
                "accuracy of the results is not guaranteed.");
    }

    if (!std::empty(model_str) && std::filesystem::exists(model_str)) {
        DORADO_LOG(debug, "Resolved model from user-specified path: '{}'", model_str);
        model_dir = model_str;

    } else if (count_model_hits(models::variant_models(), model_str) == 1) {
        const std::string& model_name = model_str;
        DORADO_LOG(debug, "Resolved model from user-specified smallvar model name: '{}'",
                   model_name);
        model_downloader::ModelDownloader downloader(models_directory, false);
        model_dir = downloader.get(model_name, "smallvar");

    } else if (count_model_hits(models::simplex_models(), model_str) == 1) {
        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0
        const std::string& basecaller_model = model_str;

        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.2.0_smallvar@v1.0
        const std::string model_name = determine_model_name(basecaller_model, false);

        DORADO_LOG(debug, "Resolved model from user-specified basecaller model name: '{}'",
                   model_name);
        model_downloader::ModelDownloader downloader(models_directory, false);
        model_dir = downloader.get(model_name, "smallvar");

    } else {
        throw std::runtime_error{"Could not resolve model from string: '" + model_str + "'."};
    }

    return model_dir;
}

secondary::ModelConfig load_model(const std::filesystem::path& model_dir,
                                  const bool load_scripted_model) {
    // Load the model.
    DORADO_LOG(info, "Parsing the model config: '{}'", (model_dir / "config.toml").string());
    const std::string model_file = load_scripted_model ? "model.pt" : "weights.pt";
    return secondary::parse_model_config(model_dir / "config.toml", model_file);
}

void validate_bam_model(const secondary::BamInfo& bam_info,
                        const secondary::ModelConfig& model_config,
                        const bool any_model,
                        const secondary::LabelSchemeType expected_label_scheme) {
    // Check that both the model and data have dwells, or that they both do not have dwells.
    const auto it_dwells = model_config.model_kwargs.find("use_dwells");
    const bool model_uses_dwells = (it_dwells != std::end(model_config.model_kwargs))
                                           ? (it_dwells->second == "true")
                                           : false;

    const bool label_scheme_is_compatible =
            secondary::parse_label_scheme_type(model_config.label_scheme_type) ==
            expected_label_scheme;

    const auto check_models_supported =
            [&model_config](const std::unordered_set<std::string>& basecaller_models) {
                // Every model from the input BAM needs to be supported.
                for (const std::string& model : basecaller_models) {
                    const bool valid = model_config.supported_basecallers.count(model);
                    if (!valid) {
                        return false;
                    }
                }
                return true;
            };

    // Stop if the label scheme is not compatible.
    if (!label_scheme_is_compatible) {
        throw std::runtime_error{"Incompatible model label scheme! Expected " +
                                 secondary::label_scheme_type_to_string(expected_label_scheme) +
                                 " but got " + model_config.label_scheme_type + "."};
    }

    if (!any_model) {
        // Verify that the basecaller model of the loaded config is compatible with the BAM.
        if (!check_models_supported(bam_info.basecaller_models)) {
            throw std::runtime_error{"Variant calling model is not compatible with the input BAM!"};
        }

        // Fail if the dwell information in the model and the data does not match.
        if (!bam_info.has_dwells && model_uses_dwells) {
            throw std::runtime_error{
                    "Input data does not contain move tables, but a model which requires move "
                    "tables has been chosen."};
        }

    } else {
        // Allow to use a model trained on a wrong basecaller model, but emit a warning.
        if (!check_models_supported(bam_info.basecaller_models)) {
            DORADO_LOG(
                    warn,
                    "Variant calling model is not compatible with the input BAM. This may produce "
                    "inferior results.");
        }

        // Allow to use a mismatched model, but emit a warning.
        if (!bam_info.has_dwells && model_uses_dwells) {
            DORADO_LOG(warn,
                       "Input data does not contain move tables, but a model which requires move "
                       "tables has been chosen. This may produce inferior results.");
        }
    }
}

std::unordered_map<std::string, std::vector<int64_t>> load_candidate_sites(
        const std::optional<std::filesystem::path>& candidate_variants_path) {
    if (!candidate_variants_path) {
        return {};
    }
    std::unordered_map<std::string, std::vector<int64_t>> ret;
    std::ifstream ifs(*candidate_variants_path);
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string chr;
        int64_t pos = 0;
        iss >> chr >> pos;
        ret[chr].emplace_back(pos);
    }
    for (auto& [chr, positions] : ret) {
        std::sort(std::begin(positions), std::end(positions));
    }
    return ret;
}

std::optional<std::unordered_map<int32_t, secondary::IntervalTreeInt64>>
create_candidate_interval_trees(
        const std::unordered_map<std::string, std::vector<int64_t>>& candidate_sites,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup) {
    std::unordered_map<int32_t, secondary::IntervalTreeInt64> trees;
    for (const auto& [ref_name, positions] : candidate_sites) {
        const auto it = draft_lookup.find(ref_name);
        if (it == std::cend(draft_lookup)) {
            DORADO_LOG(
                    warn,
                    "Candidate variant reference name '{}' not found in the input reference file! "
                    "Skipping candidates for this reference.",
                    ref_name);
            continue;
        }
        const int32_t seq_id = static_cast<int32_t>(it->second.first);
        std::vector<interval_tree::Interval<int64_t, int64_t>> intervals;
        for (const int64_t pos : positions) {
            intervals.emplace_back(pos, pos + 1, 0);
        }
        trees[seq_id] = secondary::IntervalTreeInt64(std::move(intervals));
    }
    return std::optional<std::unordered_map<int32_t, secondary::IntervalTreeInt64>>(
            std::move(trees));
}

/**
 * \brief Determine the input regions for processing. IF user_regions were provided, use those
 *          but validate them vs the input reference lookup.
 *          Otherwise, use the reference sequences listed in the input BAM file (bam_ref_seqs).
 */
std::vector<std::vector<secondary::Region>> resolve_input_regions(
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& ref_lookup,
        const std::vector<std::pair<std::string, int64_t>>& bam_ref_seqs,
        const std::vector<secondary::Region>& user_regions) {
    // Outer vector: ID of the draft, inner vector: regions.
    std::vector<std::vector<secondary::Region>> ret(std::size(ref_lookup));

    if (std::empty(user_regions)) {
        // Add full draft sequences referenced in the input BAM.
        for (const auto& [ref_name, ref_len_from_bam] : bam_ref_seqs) {
            const auto it = ref_lookup.find(ref_name);
            if (it == std::cend(ref_lookup)) {
                throw std::runtime_error{
                        "BAM header references a sequence which is not present in the input "
                        "reference FASTA file. Sequence name: '" +
                        ref_name + "'"};
            }
            const auto [ref_id, ref_len] = it->second;
            if (ref_len != ref_len_from_bam) {
                throw std::runtime_error{
                        "Length of the reference sequence differs between the input reference "
                        "FASTA and the BAM header. Sequence name: '" +
                        ref_name + "', length from FASTA: " + std::to_string(ref_len) +
                        ", length from BAM: " + std::to_string(ref_len_from_bam)};
            }
            ret[ref_id].emplace_back(secondary::Region{ref_name, 0, ref_len});
        }

    } else {
        // Bin the user regions for individual contigs.
        for (const auto& region : user_regions) {
            const auto it = ref_lookup.find(region.name);
            if (it == std::cend(ref_lookup)) {
                throw std::runtime_error(
                        "Sequence name from a custom specified region not found in the input "
                        "sequence file! region: " +
                        to_string(region));
            }
            const auto [ref_id, ref_len] = it->second;
            ret[ref_id].emplace_back(secondary::Region{region.name, region.start, region.end});
        }
    }

    return ret;
}

std::vector<std::string> load_reference_sequences(
        const std::filesystem::path& in_ref_fastx_fn,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<std::vector<secondary::Region>>& input_regions) {
    if (std::size(draft_lens) != std::size(input_regions)) {
        throw std::runtime_error{
                "Cannot load reference sequences because draft_lens and input_regions have "
                "different sizes. draft_lens.size = " +
                std::to_string(std::size(draft_lens)) +
                ", input_regions.size = " + std::to_string(std::size(input_regions))};
    }

    DORADO_LOG(debug, "[run_variant_calling] Loading full draft sequences.");
    hts_io::FastxRandomReader fastx_reader(in_ref_fastx_fn, true);

    // Dense vector by seq_id. Unused sequence IDs stay as empty strings.
    std::vector<std::string> draft_seqs(std::size(draft_lens));
    for (int64_t seq_id = 0; seq_id < std::ssize(draft_lens); ++seq_id) {
        if (std::empty(input_regions[seq_id])) {
            continue;
        }
        const auto& [ref_name, ref_len] = draft_lens[seq_id];
        std::string ref_seq = fastx_reader.fetch_seq(ref_name);
        const int64_t loaded_ref_len = std::ssize(ref_seq);

        if (loaded_ref_len != ref_len) {
            throw std::runtime_error{
                    "Length of the reference sequence differs between the input reference FASTA "
                    "index and loaded sequence. Sequence name: '" +
                    ref_name + "', length from FASTA index: " + std::to_string(ref_len) +
                    ", loaded sequence length: " + std::to_string(loaded_ref_len)};
        }

        draft_seqs[seq_id] = std::move(ref_seq);
    }

    return draft_seqs;
}

void init_progress_tracker(secondary::Stats& stats,
                           const std::vector<std::vector<secondary::Region>>& input_regions) {
    int64_t total_input_bases = std::accumulate(
            std::cbegin(input_regions), std::cend(input_regions), static_cast<int64_t>(0),
            [](const int64_t a, const std::vector<secondary::Region>& b) {
                int64_t sum = 0;
                for (const auto& region : b) {
                    sum += region.end - region.start;
                }
                return a + sum;
            });

    // Multiply by 2 because we will count each base twice for progress:
    //  1. When the inference is done.
    //  2. When the chromosome is done.
    // This ameliorates issues when some regions have zero inference samples.
    total_input_bases *= 2;

    stats.set("total", static_cast<double>(total_input_bases));
    stats.set("processed", 0.0);
}

void run_variant_calling(const Options& opt,
                         const std::string_view command_line,
                         const secondary::BamInfo& bam_info,
                         const secondary::ModelConfig& model_config,
                         smallvar::VariantResources& resources,
                         secondary::Stats& stats) {
    DORADO_LOG(info, "Threads: {}, inference threads: {}, number of devices: {}", opt.threads,
               opt.infer_threads, std::size(resources.devices));

    at::InferenceMode infer_guard;

    // Resolve the windowing parameters from either the model (default) or the CLI.
    const int32_t window_len = opt.window_len.value_or(model_config.chunk_size);
    const int32_t window_overlap = opt.window_overlap.value_or(model_config.chunk_overlap);
    const int32_t variant_flanking_bases =
            opt.variant_flanking_bases.value_or(model_config.chunk_overlap);
    const bool candidate_filtering =
            opt.candidate_filtering.value_or(model_config.candidate_filtering);
    const secondary::VariantCandidateSource variant_candidate_source =
            (!candidate_filtering)          ? secondary::VariantCandidateSource::NONE
            : (opt.candidate_variants_path) ? secondary::VariantCandidateSource::FILE
                                            : secondary::VariantCandidateSource::COMPUTE;

    if (!candidate_filtering && opt.candidate_variants_path) {
        throw std::runtime_error{
                "Candidate variants path is specified as a source ('--candidates'), but candidate "
                "filtering is not turned on by either '--candidate-filtering' or the model "
                "config."};
    }

    // Validate the final windowing parameters.
    if (window_len <= 0) {
        throw std::runtime_error{"Window size should be > 0. Given: " + std::to_string(window_len)};
    }
    if (variant_flanking_bases < 0) {
        throw std::runtime_error{"Variant flanking bases should be >= 0. Given: " +
                                 std::to_string(variant_flanking_bases)};
    }
    if ((window_overlap < 0) || (window_overlap >= window_len)) {
        throw std::runtime_error{
                "Window overlap should be >= 0 and < window_len. Given: window_overlap = " +
                std::to_string(window_overlap) + ", window_len = " + std::to_string(window_len)};
    }

    DORADO_LOG(debug,
               "Using windowing parameters: window_len = {}, window_overlap = {}, "
               "variant_flanking_bases = {}",
               window_len, window_overlap, variant_flanking_bases);
    DORADO_LOG(debug, "Using candidate region filtering: {}, source = {}", candidate_filtering,
               secondary::variant_region_source_to_string(variant_candidate_source));

    // Create a .fai index if it doesn't exist.
    const bool rv_fai = utils::create_fai_index(opt.in_ref_fastx_fn);
    if (!rv_fai) {
        DORADO_LOG(error, "Failed to create/verify a .fai index for input file: '{}'!",
                   opt.in_ref_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    // Load sequence lengths.
    DORADO_LOG(debug, "[run_variant_calling] Loading draft sequence lengths.");
    const std::vector<std::pair<std::string, int64_t>> draft_lens =
            utils::load_seq_lengths(opt.in_ref_fastx_fn);

    // Create windows only for the selected regions.
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> draft_lookup;
    for (std::size_t seq_id = 0; seq_id < std::size(draft_lens); ++seq_id) {
        draft_lookup[draft_lens[seq_id].first] = {seq_id, draft_lens[seq_id].second};
    }

    secondary::validate_regions(opt.regions, draft_lens);

    // Prepare regions for processing.
    const std::vector<std::vector<secondary::Region>> input_regions =
            resolve_input_regions(draft_lookup, bam_info.ref_seqs, opt.regions);

    std::vector<std::vector<secondary::Region>> hemizygous_regions;
    if (!std::empty(opt.hemizygous_regions)) {
        hemizygous_regions = resolve_input_regions(draft_lookup, {}, opt.hemizygous_regions);
        // Sort and merge intervals
        for (auto& ref_regions : hemizygous_regions) {
            if (std::ssize(ref_regions) > 1) {
                std::sort(std::begin(ref_regions), std::end(ref_regions));

                std::vector<secondary::Region> merged_regions;
                merged_regions.reserve(std::ssize(ref_regions));
                int64_t next_start = -1;
                int64_t next_end = -1;
                for (const auto& region : ref_regions) {
                    if (region.start > next_end) {
                        if (next_end > 0) {
                            merged_regions.emplace_back(
                                    secondary::Region{region.name, next_start, next_end});
                        }
                        next_start = region.start;
                    }
                    next_end = region.end;
                    // handle special case of unbounded end, no need to consider remaining regions
                    if (region.end == -1) {
                        break;
                    }
                }
                merged_regions.emplace_back(
                        secondary::Region{ref_regions[0].name, next_start, next_end});
                ref_regions = std::move(merged_regions);
            }
            for (const auto& region : ref_regions) {
                DORADO_LOG(debug, "Added {} to hemizygous regions", secondary::to_string(region));
            }
        }
    }

    // Load only reference sequences which are needed for the selected regions.
    const std::vector<std::string> draft_seqs =
            load_reference_sequences(opt.in_ref_fastx_fn, draft_lens, input_regions);

    // Parse candidate variant regions if a candidate file path was provided.
    const std::unordered_map<std::string, std::vector<int64_t>> candidate_sites =
            load_candidate_sites(opt.candidate_variants_path);

    // Create interval trees from candidate variant locations.
    // The variant_candidate_source can be NONE, which means no candidate filtering is applied.
    const std::optional<secondary::IntervalTreesInt64Map> candidate_trees_from_file =
            (variant_candidate_source == secondary::VariantCandidateSource::FILE)
                    ? create_candidate_interval_trees(candidate_sites, draft_lookup)
                    : std::nullopt;

    // Open the draft FASTA file. One reader per thread.
    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> draft_readers;
    draft_readers.reserve(opt.threads);
    for (int32_t i = 0; i < opt.threads; ++i) {
        draft_readers.emplace_back(
                std::make_unique<hts_io::FastxRandomReader>(opt.in_ref_fastx_fn, true));
    }
    if (std::empty(draft_readers)) {
        throw std::runtime_error("Could not create draft readers!");
    }

    // Create the output folder if needed.
    if (!std::empty(opt.output_dir)) {
        // Check if the path exists, but fail if it is not a directory.
        if (std::filesystem::exists(opt.output_dir) &&
            !std::filesystem::is_directory(opt.output_dir)) {
            throw std::runtime_error(
                    "Path specified as output directory exists, but it is not a directory: '" +
                    opt.output_dir.string() + "'.");
        }

        // Create the directory if needed/possible.
        std::filesystem::create_directories(opt.output_dir);
    }

    // Open the output stream to a file/stdout for the variant calls.
    const std::filesystem::path out_vcf_fn =
            (std::empty(opt.output_dir)) ? "-" : (opt.output_dir / "variants.vcf.gz");

    std::ofstream ofs_regions;
    if (!std::empty(opt.output_dir)) {
        const std::filesystem::path out_regions_fn = opt.output_dir / "processed_regions.bed";
        ofs_regions = std::ofstream(out_regions_fn);
    }

    // Initialize the header of the output VCF file.
    // These are the only available FILTER options.
    const secondary::Filters::FilterType vcf_filters =
            secondary::Filters::Pass | secondary::Filters::LowQual |
            secondary::Filters::IncompatZyg | secondary::Filters::NonVariant |
            secondary::Filters::LowGQ | secondary::Filters::LowAQ | secondary::Filters::LowVQ;

    // Set the sample name for VCF output
    // If opt.sample_name has not been set explicitly, try to set sample name from BAM.
    // If the BAM contains multiple sample names, exit and ask for user instruction.
    const std::string& sample_name = [&opt, &bam_info]() {
        if (opt.sample_name.has_value()) {
            return opt.sample_name.value();
        }
        if (bam_info.sample_names.empty()) {
            return secondary::VCF_DEFAULT_SAMPLE;
        }
        return unique_sample_name_or_throw(bam_info);
    }();
    DORADO_LOG(info, "Setting sample name for VCF output: {}", sample_name);

    // VCF writer, nullptr unless variant calling is run.
    std::unique_ptr<secondary::VCFWriter> vcf_writer = std::make_unique<secondary::VCFWriter>(
            out_vcf_fn, vcf_filters, draft_lens, opt.out_format == VariantCallingFormatEnum::GVCF,
            sample_name, command_line);

    // Optionally write Kadayashi variants.
    std::optional<secondary::VCFWriter> vcf_writer_kadayashi;
    std::optional<secondary::VCFWriter> vcf_writer_inference;
    if (candidate_filtering && opt.dump_variants) {
        const std::string out_vcf_kadayashi_fn =
                (std::empty(opt.output_dir)) ? "-" : (opt.output_dir / "kadayashi.vcf.gz").string();
        vcf_writer_kadayashi.emplace(out_vcf_kadayashi_fn, vcf_filters, draft_lens, false,
                                     sample_name, command_line);

        const std::string out_vcf_inference_fn =
                (std::empty(opt.output_dir)) ? "-" : (opt.output_dir / "inference.vcf.gz").string();
        vcf_writer_inference.emplace(out_vcf_inference_fn, vcf_filters, draft_lens,
                                     opt.out_format == VariantCallingFormatEnum::GVCF, sample_name,
                                     command_line);
    }

    // Compute the minimum usable memory across all devices and use that as the batch size.
    // Reason: batches are constructed and pushed to a queue, workers only pop the batches from
    // the queue, and minimum possible batch size needs to be satisfied.
    const double min_avail_mem = [&resources]() {
        if (std::empty(resources.devices)) {
            return 0.0;
        }
        double ret = resources.devices.front().available_memory_GB;
        for (const secondary::DeviceInfo& device_info : resources.devices) {
            ret = std::min(ret, device_info.available_memory_GB);
        }
        return ret;
    }();
    constexpr double AVAILABLE_MEMORY_FACTOR = 0.85;
    const double usable_mem = (min_avail_mem * AVAILABLE_MEMORY_FACTOR) / opt.infer_threads;

    if (opt.batch_size > 0) {
        DORADO_LOG(info, "Using fixed batch size: {}", opt.batch_size);
    } else {
        DORADO_LOG(info, "Using auto computed batch size. Usable per-worker memory: {:.2f} GB",
                   usable_mem);
    }

    const int32_t ploidy = secondary::label_scheme_type_to_ploidy(
            secondary::parse_label_scheme_type(model_config.label_scheme_type));

    init_progress_tracker(stats, input_regions);

    const int32_t flank_trim_len =
            (variant_candidate_source == secondary::VariantCandidateSource::FILE)
                    ? 0
                    : opt.flank_trim_len;

    // Create the BAM regions per chromosome.
    std::vector<std::vector<secondary::Window>> bam_regions;
    bam_regions.reserve(std::ssize(input_regions));
    {
        for (const auto& ref_regions : input_regions) {
            std::vector<secondary::Window> new_bam_regions = secondary::create_windows_from_regions(
                    ref_regions, draft_lookup, opt.bam_chunk, window_overlap);
            bam_regions.emplace_back(std::move(new_bam_regions));
        }
    }

    const QueueSizes queue_sizes = resolve_queue_sizes(opt, std::size(resources.models));
    DORADO_LOG(debug,
               "Queue sizes: bam_region = {}, sample = {}, batch = {}, decode = {}, vc_data = {}, "
               "vc_writer = {}",
               queue_sizes.bam_region, queue_sizes.sample, queue_sizes.batch, queue_sizes.decode,
               queue_sizes.vc_data, queue_sizes.vc_writer);

    utils::AsyncQueue<secondary::Window> bam_region_queue(queue_sizes.bam_region);
    utils::AsyncQueue<smallvar::InferenceData> sample_queue(queue_sizes.sample);
    utils::AsyncQueue<smallvar::InferenceData> batch_queue(queue_sizes.batch);
    utils::AsyncQueue<smallvar::DecodeData> decode_queue(queue_sizes.decode);
    utils::AsyncQueue<secondary::VariantCallingSample> vc_data_queue(queue_sizes.vc_data);
    utils::AsyncQueue<int64_t> vc_writer_queue(queue_sizes.vc_writer);

    // Initialize data needed to reduce the processing (decode/merge/trim results).
    std::vector<smallvar::ChromosomeReduceData> chrom_reduce_data(std::size(input_regions));
    {
        for (int64_t seq_id = 0; seq_id < std::ssize(bam_regions); ++seq_id) {
            auto& crd = chrom_reduce_data[seq_id];
            crd.seq_id = seq_id;
            std::tie(crd.seq_name, crd.seq_len) = draft_lens[seq_id];
            crd.num_bam_regions = std::ssize(bam_regions[seq_id]);
            crd.remaining_bam_regions = std::ssize(bam_regions[seq_id]);
            crd.progress_target = std::accumulate(
                    std::cbegin(bam_regions[seq_id]), std::cend(bam_regions[seq_id]),
                    static_cast<int64_t>(0), [](const int64_t sum, const secondary::Window& w) {
                        return sum + std::max<int64_t>(0, w.end_no_overlap - w.start_no_overlap);
                    });
            crd.selected_regions.reserve(std::size(input_regions[seq_id]));
            for (const secondary::Region& region : input_regions[seq_id]) {
                const secondary::RegionInt selected_region{static_cast<int32_t>(seq_id),
                                                           region.start, region.end};
                if (!secondary::is_valid(selected_region)) {
                    continue;
                }
                const secondary::RegionInt normalized_region =
                        secondary::normalize_region(selected_region, crd.seq_len);
                crd.selected_regions.emplace_back(normalized_region);
            }

            // If this chromosome has no BAM regions to process, mark it as ready so it gets popped.
            if (std::empty(bam_regions[seq_id])) {
                chrom_reduce_data[seq_id].ready = true;
            }
        }
    }

    // Capture any possible exceptions from worker threads in these objects.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus wrs_sample_producer;
    secondary::WorkerReturnStatus wrs_batch_producer;
    secondary::WorkerReturnStatus wrs_infer;
    secondary::WorkerReturnStatus wrs_separate_infer_output;
    secondary::WorkerReturnStatus wrs_thread_call_variants;
    secondary::WorkerReturnStatus wrs_thread_write_variants;

    // Worker to push BAM regions to the processing queue.
    auto thread_bam_region_producer = utils::jthread([&] {
        utils::set_thread_name("bam_region_producer");
        bool terminated = false;
        for (int64_t i = 0; (i < std::ssize(bam_regions)) && !terminated; ++i) {
            for (int64_t j = 0; j < std::ssize(bam_regions[i]); ++j) {
                secondary::Window w = bam_regions[i][j];
                const auto status = bam_region_queue.try_push(std::move(w));
                if (status == utils::AsyncQueueStatus::Terminate) {
                    terminated = true;
                    break;
                }
            }
        }
        if (!worker_terminate.load(std::memory_order_acquire)) {
            bam_region_queue.terminate(utils::AsyncQueueTerminateFast::No);
        }
    });

    auto thread_queue_terminator = utils::jthread([&] {
        utils::set_thread_name("variant_queue_terminator");
        worker_terminate.wait(false, std::memory_order_acquire);
        bam_region_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
        sample_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
        batch_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
        decode_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
        vc_data_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
        vc_writer_queue.terminate(utils::AsyncQueueTerminateFast::Yes);
    });

    // Async workflow, this block joins the threads.
    {
        // Create a thread for worker_sample_producer.
        auto thread_sample_producer = utils::jthread([&] {
            utils::set_thread_name("worker_sample_producer");
            smallvar::worker_sample_producer(
                    bam_region_queue, sample_queue, chrom_reduce_data, resources, stats,
                    worker_terminate, wrs_sample_producer, bam_regions, draft_lens, draft_seqs,
                    variant_candidate_source, candidate_trees_from_file, opt.threads, window_len,
                    window_overlap, variant_flanking_bases, opt.continue_on_error, ploidy,
                    opt.pass_min_qual, opt.tiled_regions, opt.tiled_ext_flanks, opt.tiled_ext_major,
                    opt.tiled_ext_min_cov, opt.tiled_ext_cov_fract, opt.min_depth);
        });

        // Create a thread for worker_batch_producer.
        auto thread_batch_producer = utils::jthread([&] {
            utils::set_thread_name("worker_batch_producer");
            smallvar::worker_batch_producer(sample_queue, batch_queue, worker_terminate,
                                            wrs_batch_producer, *resources.models.front(),
                                            window_len, opt.batch_size, usable_mem,
                                            opt.continue_on_error);
        });

        auto thread_infer = utils::jthread([&] {
            utils::set_thread_name("worker_infer_samples_in_parallel");
            smallvar::worker_infer_samples_in_parallel(
                    batch_queue, decode_queue, resources.models, worker_terminate, wrs_infer,
                    resources.streams, resources.encoders, draft_lens, opt.continue_on_error);
        });

        auto thread_separate_infer_output = utils::jthread([&] {
            utils::set_thread_name("worker_separate_decode_data");
            const int32_t num_threads = static_cast<int32_t>(
                    std::min(std::ssize(resources.models), std::ssize(resources.encoders)));
            const secondary::FeatureColumnMap feature_column_map =
                    (!std::empty(resources.encoders))
                            ? resources.encoders.front()->get_feature_column_map()
                            : secondary::FeatureColumnMap{};
            smallvar::worker_separate_decode_data(decode_queue, vc_data_queue, worker_terminate,
                                                  wrs_separate_infer_output, feature_column_map,
                                                  num_threads, opt.continue_on_error);
        });

        auto thread_call_variants = utils::jthread([&] {
            utils::set_thread_name("worker_variant_calling_reduce");
            smallvar::worker_variant_calling_reduce(
                    vc_data_queue, vc_writer_queue, chrom_reduce_data, worker_terminate,
                    wrs_thread_call_variants, stats, draft_readers, opt.continue_on_error,
                    opt.threads, draft_lens, *resources.decoder, hemizygous_regions,
                    opt.pass_min_qual, opt.pass_min_gq, opt.pass_min_aq, opt.pass_min_vq,
                    opt.ambig_ref, opt.out_format == VariantCallingFormatEnum::GVCF, ploidy,
                    flank_trim_len, variant_candidate_source);
        });

        auto thread_write_variants = utils::jthread([&] {
            utils::set_thread_name("worker_variant_writer");
            smallvar::worker_variant_writer(vc_writer_queue, chrom_reduce_data, worker_terminate,
                                            wrs_thread_write_variants, *vcf_writer, ofs_regions,
                                            vcf_writer_kadayashi, vcf_writer_inference,
                                            opt.continue_on_error);
        });
    }

    smallvar::signal_worker_terminate(worker_terminate);

    // Propagate exceptions from threads.
    for (const auto& wrs :
         {wrs_sample_producer, wrs_batch_producer, wrs_infer, wrs_separate_infer_output,
          wrs_thread_call_variants, wrs_thread_write_variants}) {
        if (wrs.exception_thrown) {
            throw std::runtime_error{wrs.message};
        }
    }
}

}  // namespace

int small_variant_caller(int argc, char* argv[]) {
    try {
        const std::vector<std::string_view> arguments(argv, argv + argc);
        const std::string command_line = utils::join(arguments, " ");

        // Initialize CLI options. The parse_args below requires a non-const reference.
        // Verbosity is passed into a callback, so we need it here.
        int verbosity = 0;
        argparse::ArgumentParser parser("dorado smallvar", DORADO_VERSION,
                                        argparse::default_arguments::help);
        add_arguments(parser, verbosity);

        // Parse the arguments.
        const int rv_parse = parse_args(argc, argv, parser);

        if (rv_parse != EXIT_SUCCESS) {
            return rv_parse;
        }

        // Initialize the options from the CLI.
        const Options opt = set_options(parser, verbosity);

        if (opt.verbosity == 0) {
            spdlog::set_level(spdlog::level::info);
        } else if (opt.verbosity == 1) {
            spdlog::set_level(spdlog::level::debug);
        } else if (opt.verbosity >= 2) {
            spdlog::set_level(spdlog::level::trace);
        }

        spdlog::flush_every(std::chrono::seconds(1));

        // Check if input options are good.
        validate_options(opt);

#if DORADO_CUDA_BUILD
        cli::log_requested_cuda_devices(opt.device_str);
#endif

        // Get info from BAM needed for the run.
        const secondary::BamInfo bam_info =
                secondary::analyze_bam(opt.in_aln_bam_fn, opt.read_group);

        // Debug printing.
        {
            DORADO_LOG(debug, "bam_info.uses_dorado_aligner = {}", bam_info.uses_dorado_aligner);
            DORADO_LOG(debug, "bam_info.has_dwells = {}", bam_info.has_dwells);
            DORADO_LOG(debug, "bam_info.read_groups:");
            for (const auto& rg : bam_info.read_groups) {
                DORADO_LOG(debug, "    - {}", rg);
            }
            DORADO_LOG(debug, "bam_info.basecaller_models:");
            for (const auto& rg : bam_info.basecaller_models) {
                DORADO_LOG(debug, "    - {}", rg);
            }
            DORADO_LOG(debug, "bam_info.sample_names:");
            for (const auto& rg : bam_info.sample_names) {
                DORADO_LOG(debug, "    - {}", rg);
            }
            if (!std::empty(opt.read_group)) {
                DORADO_LOG(debug,
                           "Only the user-requested RG was selected from the input bam. RG: '{}'",
                           opt.read_group);
            }
        }

        // Allow only Dorado aligned BAMs.
        if ((bam_info.uses_dorado_aligner == false) && (opt.any_bam == false)) {
            throw std::runtime_error("Input BAM file was not aligned using Dorado.");
        }

        // Validate the read groups in the BAM file.
        if (!std::empty(opt.read_group) || !opt.ignore_read_groups) {
            secondary::check_read_groups(bam_info, opt.read_group);
        }

        // Set the number of threads so that libtorch doesn't cause a thread bomb.
        utils::initialise_torch();

        // Resolve the model.
        secondary::ModelConfig model_config;
        if (std::empty(opt.model_str)) {
            constexpr bool ANY_MODEL = false;
            // Basic mainstream model resolving.
            const std::filesystem::path model_dir = resolve_model(bam_info, opt.models_directory);
            model_config = load_model(model_dir, opt.load_scripted_model);
            validate_bam_model(bam_info, model_config, ANY_MODEL,
                               secondary::LabelSchemeType::DIPLOID);
        } else {
            constexpr bool ANY_MODEL = true;
            // Advanced model resolve from a specific path or model name.
            const std::filesystem::path model_dir = resolve_model_manually(
                    bam_info, opt.models_directory, opt.model_str, ANY_MODEL);
            model_config = load_model(model_dir, opt.load_scripted_model);
            validate_bam_model(bam_info, model_config, ANY_MODEL,
                               secondary::LabelSchemeType::DIPLOID);
        }

        // Create the models, encoders and BAM handles.
        smallvar::VariantResources resources = smallvar::create_resources(
                model_config, opt.in_ref_fastx_fn, opt.in_aln_bam_fn, opt.device_str, opt.threads,
                opt.infer_threads, opt.full_precision, opt.read_group, opt.tag_name, opt.tag_value,
                opt.min_snp_accuracy, opt.tag_keep_missing, opt.min_mapq, opt.haplotag_source,
                opt.phasing_bin_path, opt.kadayashi_opt, opt.legacy_feature_gen);

        // Progress bar.
        secondary::Stats stats;
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        smallvar::VariantProgressTracker tracker;
        std::vector<dorado::stats::StatsCallable> stats_callables;
        stats_callables.push_back([&tracker, &stats](const stats::NamedStats& /*stats*/) {
            tracker.update_progress_bar(stats.get_stats());
        });
        constexpr auto kStatsPeriod = std::chrono::milliseconds(1000);
        auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));

        // Log that that half-precision will be used.
        if (!opt.full_precision) {
            DORADO_LOG(info, "Using half precision!");
        }

        run_variant_calling(opt, command_line, bam_info, model_config, resources, stats);

        tracker.finalize();
        stats_sampler->terminate();

        // Hack to clear the last line from the progress bar. The library automatically does '\r'.
        std::cerr << std::string(200, ' ') << '\r';
        DORADO_LOG(info, "Done!");

    } catch (const std::exception& e) {
        DORADO_LOG(error, e.what());
        return EXIT_FAILURE;
    } catch (...) {
        DORADO_LOG(error, "Caught an unknown exception!");
        return EXIT_FAILURE;
    }

    return 0;
}

}  // namespace dorado
