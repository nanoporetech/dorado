#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "dorado_version.h"
#include "hts_utils/FastxRandomReader.h"
#include "hts_utils/fai_utils.h"
#include "model_downloader/model_downloader.h"
#include "model_resolution.h"
#include "models/models.h"
#include "polish/polish_impl.h"
#include "polish_progress_tracker.h"
#include "secondary/architectures/model_config.h"
#include "secondary/common/bam_info.h"
#include "secondary/common/batching.h"
#include "secondary/common/stats.h"
#include "secondary/common/vcf_writer.h"
#include "secondary/consensus/variant_calling.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/torch_utils.h"
#include "utils/AsyncQueue.h"
#include "utils/arg_parse_ext.h"
#include "utils/fs_utils.h"
#include "utils/io_utils.h"
#include "utils/jthread.h"
#include "utils/log_utils.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"
#include "utils/thread_utils.h"

#include <ATen/Parallel.h>
#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

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

#ifndef _WIN32
#include <unistd.h>
#endif

// #define DEBUG_POLISH_DUMP_SEQ_PIECES
// #define DEBUG_POLISH_REGIONS
// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado {

namespace {

enum class OutputFormat {
    FASTA,
    FASTQ,
};

enum class VariantCallingEnum {
    VCF,
    GVCF,
};

/// \brief All options for this tool.
struct Options {
    // Positional parameters.
    std::filesystem::path in_aln_bam_fn;
    std::filesystem::path in_draft_fastx_fn;

    // Optional parameters.
    std::filesystem::path output_dir;
    std::string model_str;
    std::optional<std::filesystem::path> models_directory;
    OutputFormat out_format = OutputFormat::FASTA;
    int32_t verbosity = 0;
    int32_t threads = 0;
    int32_t infer_threads = 1;
    std::string device_str;
    int32_t batch_size = 16;
    int64_t draft_batch_size = 200'000'000;
    int32_t window_len = 10000;
    int32_t window_overlap = 1000;
    int32_t bam_chunk = 1'000'000;
    int32_t bam_subchunk = 100'000;
    std::optional<std::string> regions_str;
    std::vector<secondary::Region> regions;
    bool full_precision = false;
    bool load_scripted_model = false;
    int32_t queue_size = 1000;
    bool fill_gaps = true;
    std::optional<char> fill_char;
    std::optional<int32_t> min_mapq;
    std::string read_group;
    bool ignore_read_groups = false;
    std::string tag_name;
    int32_t tag_value = 0;
    std::optional<bool> tag_keep_missing;  // Optionally overrides the model config if specified.
    int32_t min_depth = 0;
    bool any_bam = false;
    bool any_model = false;
    bool bacteria = false;
    VariantCallingEnum vc_type = VariantCallingEnum::VCF;
    bool ambig_ref = false;
    bool run_variant_calling = false;
    bool write_consensus = false;
    bool continue_on_error = false;
};

/// \brief Define the CLI options.
void add_arguments(argparse::ArgumentParser& parser, int& verbosity) {
    parser.add_description("Consensus tool for polishing draft assemblies");

    {
        // Positional arguments group
        parser.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser.add_argument("in_draft_fastx").help("Draft assembly for polishing");
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
                                  "'cuda:<device_id>[,<device_id>...]'. Specifying 'auto' will "
                                  "choose either 'cpu' "
                                  "or 'cuda:all' depending on the presence of a GPU device."})
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
        parser.add_argument("--bacteria")
                .help("Optimise polishing for plasmids and bacterial genomes.")
                .flag();
        parser.add_argument("-q", "--qualities")
                .help("Output with per-base quality scores (FASTQ).")
                .flag();
        parser.add_argument("--vcf")
                .help("Output a VCF file with variant calls to --output-dir if specified, "
                      "otherwise to stdout.")
                .flag();
        parser.add_argument("--gvcf")
                .help("Output a gVCF file to --output-dir if specified, otherwise to stdout.")
                .flag();
        parser.add_argument("--ambig-ref")
                .help("Decode variants at ambiguous reference positions.")
                .flag();
    }
    {
        parser.add_group("Advanced options");
        parser.add_argument("-b", "--batchsize")
                .help("Batch size for inference.")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("--draft-batchsize")
                .help("Approximate batch size for processing input draft sequences.")
                .default_value(std::string{"200M"});
        parser.add_argument("--window-len")
                .help("Window size for calling consensus.")
                .default_value(10000)
                .scan<'i', int>();
        parser.add_argument("--window-overlap")
                .help("Overlap length between windows.")
                .default_value(1000)
                .scan<'i', int>();
        parser.add_argument("--bam-chunk")
                .help("Size of draft chunks to parse from the input BAM at a time.")
                .default_value(1000000)
                .scan<'i', int>();
        parser.add_argument("--bam-subchunk")
                .help("Size of regions to split the bam_chunk in to for parallel processing")
                .default_value(100000)
                .scan<'i', int>();
        parser.add_argument("--no-fill-gaps")
                .help("Do not fill gaps in consensus sequence with draft sequence.")
                .flag();
        parser.add_argument("--fill-char").help("Use a designated character to fill gaps.");
        parser.add_argument("--regions")
                .help("Process only these regions of the input. Can be either a path to a BED file "
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
                .default_value(0)
                .scan<'i', int>();
    }

    // Hidden advanced arguments.
    {
        parser.add_argument("--full-precision")
                .hidden()
                .help("Always use full precision for inference.")
                .flag();
        parser.add_argument("--queue-size")
                .hidden()
                .help("Queue size for processing.")
                .default_value(1000)
                .scan<'i', int>();
        parser.add_argument("--scripted")
                .hidden()
                .help("Load the scripted Torch model instead of building one internally.")
                .flag();
        parser.add_argument("--any-bam")
                .hidden()
                .help("Allow any BAM as input, not just Dorado aligned.")
                .flag();
        parser.add_argument("--skip-model-compatibility-check")
                .hidden()
                .help("Allow any model to be applied on the data.")
                .flag();
        parser.add_argument("--continue-on-error")
                .hidden()
                .help("Continue the process even if an exception is thrown. This "
                      "may leave some regions unprocessed.")
                .flag();
        parser.add_argument("--model-override")
                .hidden()
                .help("Path to a model folder or an exact name of a model to use.")
                .default_value("");
    }
}

int parse_args(int argc, char** argv, argparse::ArgumentParser& parser) {
    try {
        cli::parse(parser, argc, argv);

    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/// \brief This function simply fills out the Options struct with the parsed CLI args.
Options set_options(const argparse::ArgumentParser& parser, const int verbosity) {
    Options opt;

    opt.in_aln_bam_fn = parser.get<std::string>("in_aln_bam");
    opt.in_draft_fastx_fn = parser.get<std::string>("in_draft_fastx");

    opt.output_dir = parser.get<std::string>("output-dir");
    opt.bacteria = parser.get<bool>("bacteria");

    opt.models_directory = model_resolution::get_models_directory(parser);

    opt.out_format = parser.get<bool>("qualities") ? OutputFormat::FASTQ : OutputFormat::FASTA;
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
    opt.draft_batch_size =
            std::max<int64_t>(0, utils::arg_parse::parse_string_to_size<int64_t>(
                                         parser.get<std::string>("draft-batchsize")));
    opt.window_len = parser.get<int>("window-len");
    opt.window_overlap = parser.get<int>("window-overlap");
    opt.bam_chunk = parser.get<int>("bam-chunk");
    opt.bam_subchunk = parser.get<int>("bam-subchunk");
    opt.verbosity = verbosity;
    opt.regions_str = parser.present<std::string>("regions");
    if (opt.regions_str) {
        opt.regions = secondary::parse_regions(*opt.regions_str);
    }
    opt.min_depth = parser.get<int>("min-depth");

    opt.full_precision = parser.get<bool>("full-precision");
    opt.load_scripted_model = parser.get<bool>("scripted");
    opt.queue_size = parser.get<int>("queue-size");
    opt.any_bam = parser.get<bool>("any-bam");
    opt.any_model = parser.get<bool>("skip-model-compatibility-check");
    opt.continue_on_error = parser.get<bool>("continue-on-error");
    opt.model_str = parser.get<std::string>("model-override");

    opt.fill_gaps = !parser.get<bool>("no-fill-gaps");
    opt.fill_char = (parser.is_used("--fill-char"))
                            ? std::optional<char>{parser.get<std::string>("fill-char")[0]}
                            : std::nullopt;
    opt.read_group = (parser.is_used("--RG")) ? parser.get<std::string>("RG") : "";
    opt.ignore_read_groups = parser.get<bool>("ignore-read-groups");
    opt.tag_name = parser.get<std::string>("tag-name");
    opt.tag_value = parser.get<int>("tag-value");
    // The `"--tag-keep-missing` is a special case because it's a flag, and we cannot use `present`.
    opt.tag_keep_missing = (parser.is_used("--tag-keep-missing"))
                                   ? std::optional<bool>{parser.get<bool>("tag-keep-missing")}
                                   : std::nullopt;
    opt.min_mapq = parser.present<int32_t>("min-mapq");

    if (opt.bam_subchunk > opt.bam_chunk) {
        spdlog::warn(
                "BAM sub-chunk size is larger than bam_chunk size. Limiting to bam_chunk size. "
                "bam_subchunk = {}, bam_chunk = {}",
                opt.bam_chunk, opt.bam_subchunk);
        opt.bam_subchunk = opt.bam_chunk;
    }

    // Variant calling setup.
    const bool vcf = parser.get<bool>("vcf");
    const bool gvcf = parser.get<bool>("gvcf");
    opt.run_variant_calling = false;
    if (vcf && gvcf) {
        throw std::runtime_error{
                "Both --vcf and --gvcf are specified. Only one of these options can be used."};
    } else if (vcf) {
        opt.vc_type = VariantCallingEnum::VCF;
        opt.run_variant_calling = true;
    } else if (gvcf) {
        opt.vc_type = VariantCallingEnum::GVCF;
        opt.run_variant_calling = true;
    }
    opt.ambig_ref = parser.get<bool>("ambig-ref");

    // Write the consensus sequence only if: (1) to a folder, or (2) to stdout with no VC options specified.
    if (!std::empty(opt.output_dir) || (!vcf && !gvcf)) {
        opt.write_consensus = true;
    }

    return opt;
}

void validate_options(const Options& opt) {
    // Parameter validation.
    if (!cli::validate_device_string(opt.device_str)) {
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_aln_bam_fn)) {
        spdlog::error("Input draft file {} does not exist!", opt.in_aln_bam_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_draft_fastx_fn)) {
        spdlog::error("Input reads file {} does not exist!", opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (opt.batch_size < 0) {
        spdlog::error("Batch size should be >= 0. Given: {}.", opt.batch_size);
        std::exit(EXIT_FAILURE);
    }
    if (opt.draft_batch_size <= 0) {
        spdlog::error("Draft batch size should be > 0. Given: {}.", opt.draft_batch_size);
        std::exit(EXIT_FAILURE);
    }
    if (opt.window_len <= 0) {
        spdlog::error("Window size should be > 0. Given: {}.", opt.window_len);
        std::exit(EXIT_FAILURE);
    }
    if (opt.bam_chunk <= 0) {
        spdlog::error("BAM chunk size should be > 0. Given: {}.", opt.bam_chunk);
        std::exit(EXIT_FAILURE);
    }
    if (opt.bam_subchunk <= 0) {
        spdlog::error("BAM sub-chunk size should be > 0. Given: {}.", opt.bam_chunk);
        std::exit(EXIT_FAILURE);
    }
    if ((opt.window_overlap < 0) || (opt.window_overlap >= opt.window_len)) {
        spdlog::error(
                "Window overlap should be >= 0 and < window_len. Given: window_overlap = {}, "
                "window_len = {}.",
                opt.window_overlap, opt.window_len);
        std::exit(EXIT_FAILURE);
    }

    if (!std::filesystem::exists(opt.in_aln_bam_fn) ||
        std::filesystem::is_empty(opt.in_aln_bam_fn)) {
        spdlog::error("Input file {} does not exist or is empty.", opt.in_aln_bam_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_draft_fastx_fn) ||
        std::filesystem::is_empty(opt.in_draft_fastx_fn)) {
        spdlog::error("Input file {} does not exist or is empty.", opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    if (opt.queue_size <= 0) {
        spdlog::error("Queue size needs to be > 0, given: {}.", opt.queue_size);
        std::exit(EXIT_FAILURE);
    }

    if ((std::size(opt.tag_name) == 1) || (std::size(opt.tag_name) > 2)) {
        spdlog::error(
                "The tag_name is specified, but it needs to contain exactly two characters. Given: "
                "'{}'.",
                opt.tag_name);
        std::exit(EXIT_FAILURE);
    }

    if (opt.regions_str && std::empty(opt.regions)) {
        spdlog::error("Option --regions is specified, but an empty set of regions is given!");
        std::exit(EXIT_FAILURE);
    }
}

/**
 * \brief Writes the consensus sequence from the given results. The dimensions of the `results` vector
 *          are: [part_id x haplotype_id].
 */
void write_consensus_results(std::ostream& os,
                             const std::vector<std::vector<secondary::ConsensusResult>>& results,
                             const bool fill_gaps,
                             const bool write_quals) {
    if (std::empty(results)) {
        return;
    }

    for (size_t i = 0; i < std::size(results); ++i) {
        if (std::empty(results[i])) {
            continue;
        }

        for (size_t hap_id = 0; hap_id < std::size(results[i]); ++hap_id) {
            const std::string hap_label =
                    (std::size(results[i]) > 1) ? ("_hap_" + std::to_string(hap_id)) : "";

            secondary::ConsensusResult out = results[i][hap_id];
            polisher::remove_deletions(out);

            std::string header = out.name + hap_label;

            if (!fill_gaps) {
                header += "_" + std::to_string(i) + " " + std::to_string(out.draft_start) + "-" +
                          std::to_string(out.draft_end);
            }

            if (write_quals) {
                os << '@' << header << '\n' << out.seq << "\n+\n" << out.quals << '\n';
            } else {
                os << '>' << header << '\n' << out.seq << '\n';
            }
        }
    }
}

// clang-format off
// Look-up table from a basecaller model to a legacy polish model (base)name.
const std::unordered_map<std::string, std::string> lut_basecaller_to_legacy_polish_model{
    {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_polish"},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_polish"},
    {"dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish"},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.3.0", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish"},
};
// Look-up table from a basecaller model to a polish model (base)name.
const std::unordered_map<std::string, std::string> lut_basecaller_to_polish_model{
    {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl"},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.0.0", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_polish_rl"},
    {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0", "dna_r10.4.1_e8.2_400bps_hac@v5.2.0_polish_rl"},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.2.0", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0_polish_rl"},
};
// Look-up table from a basecaller model to a bacterial model name.
const std::unordered_map<std::string, std::string> lut_basecaller_to_bacterial_model{
    {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.3.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.0.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
};
const std::unordered_set<std::string> legacy_basecaller_models{
    "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
};
// Set of all basecaller models supported by polish.
const std::unordered_set<std::string> all_basecaller_models{
    "dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v5.2.0", "dna_r10.4.1_e8.2_400bps_sup@v5.2.0",
};
// Look-up table of basecaller model compatibilities for the bacterial models.
// E.g. bacterial model for `dna_r10.4.1_e8.2_400bps_hac@v5.2.0` is fully compatible (and at the moment
// identical) to `dna_r10.4.1_e8.2_400bps_hac@v5.0.0`, `dna_r10.4.1_e8.2_400bps_sup@v4.3.0`, etc.
// This is important for forward compatibility of existing models to avoid having to create a new copy of an
// existing model every time a new basecaller model is released and the polishing model stays the same.
const std::unordered_map<std::string, std::unordered_set<std::string>> lut_compatible_basecallers_for_bacterial{
    {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_hac@v4.3.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_sup@v4.3.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.0.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_hac@v5.2.0", all_basecaller_models},
    {"dna_r10.4.1_e8.2_400bps_sup@v5.2.0", all_basecaller_models},
};
// clang-format on

std::unordered_set<std::string> get_compatible_model_set(const std::string& basecaller_model,
                                                         const bool bacteria) {
    // Bacterial models can be compatible with multiple basecaller models.
    if (bacteria) {
        const auto it = lut_compatible_basecallers_for_bacterial.find(basecaller_model);
        if (it == std::cend(lut_compatible_basecallers_for_bacterial)) {
            return {};
        }
        return it->second;
    }
    // Non-bacterial models are compatible with only one basecaller model.
    return {basecaller_model};
}

bool sets_intersect(const std::unordered_set<std::string>& set1,
                    const std::unordered_set<std::string>& set2) {
    for (const std::string& val : set1) {
        if (set2.count(val) > 0) {
            return true;
        }
    }
    return false;
}

std::string determine_model_name(const std::string& basecaller_model,
                                 const bool bacteria,
                                 const bool has_dwells) {
    // Return a bacterial model if requested.
    if (bacteria) {
        // Resolve a bacterial model.
        const auto it = lut_basecaller_to_bacterial_model.find(basecaller_model);
        if (it == std::cend(lut_basecaller_to_bacterial_model)) {
            throw std::runtime_error("There are no bacterial models for the basecaller model: '" +
                                     basecaller_model + "'.");
        }
        // Found a bacterial polishing model.
        return it->second;
    }

    // Check if this is a legacy polishing model. These don't have move table support.
    const auto it_legacy = lut_basecaller_to_legacy_polish_model.find(basecaller_model);
    if (it_legacy != std::cend(lut_basecaller_to_legacy_polish_model)) {
        // Found a legacy polishing model.
        return it_legacy->second;
    }

    const auto it = lut_basecaller_to_polish_model.find(basecaller_model);
    if (it == std::cend(lut_basecaller_to_polish_model)) {
        throw std::runtime_error("There are no polishing models for the basecaller model: '" +
                                 basecaller_model + "'.");
    }

    // Found a polishing model.
    const std::string polish_model_suffix(has_dwells ? "_mv" : "");
    return it->second + polish_model_suffix;
}

int32_t count_model_hits(const dorado::models::ModelList& model_list,
                         const std::string& model_name) {
    int32_t num_found = 0;
    for (const auto& info : model_list) {
        if (info.name == model_name) {
            ++num_found;
        }
    }
    return num_found;
}

void print_basecaller_models(std::ostream& os,
                             const std::unordered_set<std::string>& basecaller_models,
                             const std::string& delimiter) {
    std::vector<std::string> lines(std::begin(basecaller_models), std::end(basecaller_models));
    std::sort(std::begin(lines), std::end(lines));
    os << utils::join(lines, delimiter);
}

const std::filesystem::path resolve_model(
        const secondary::BamInfo& bam_info,
        const std::optional<std::filesystem::path>& models_directory,
        const bool bacteria) {
    spdlog::info("Auto resolving the model.");

    // Check that there is at least one basecaller listed in the BAM. Otherwise, no auto resolving.
    if (std::size(bam_info.basecaller_models) != 1) {
        if (std::empty(bam_info.basecaller_models)) {
            throw std::runtime_error{
                    "Input BAM file has no basecaller models listed in the header."};
        }
        if (std::size(bam_info.basecaller_models) > 1) {
            std::ostringstream oss;
            oss << "Input BAM file has a mix of different basecaller models. Only one basecaller "
                   "model can be processed. List of all basecaller models found in the BAM file: ";
            print_basecaller_models(oss, bam_info.basecaller_models, ", ");
            throw std::runtime_error{oss.str()};
        }
    }

    // Check if any of the input models is a stereo, to report a clear error that this is not supported.
    for (const std::string& model : bam_info.basecaller_models) {
        if (model.find("stereo") != std::string::npos) {
            std::ostringstream oss;
            oss << "Inputs from duplex basecalling are not supported. Detected model: '" << model
                << "' in the input BAM.";
            throw std::runtime_error{oss.str()};
        }
    }

    // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0
    const std::string& basecaller_model = *std::begin(bam_info.basecaller_models);

    // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv
    const std::string model_name =
            determine_model_name(basecaller_model, bacteria, bam_info.has_dwells);

    // Sanity check that the model name exists in the polishing models.
    if (count_model_hits(models::polish_models(), model_name) == 0) {
        throw std::runtime_error{"Resolved model '" + model_name + "' not found!"};
    }

    spdlog::debug("Resolved model from input data: {}", model_name);

    model_downloader::ModelDownloader downloader(models_directory);
    const std::filesystem::path model_dir = downloader.get(model_name, "polish");

    return model_dir;
}

std::filesystem::path resolve_model_advanced(
        const secondary::BamInfo& bam_info,
        const std::optional<std::filesystem::path>& models_directory,
        const std::string& model_str,
        const bool any_model) {
    if (bam_info.has_dwells) {
        spdlog::info("Input data contains move tables.");
    } else {
        spdlog::info("Input data does not contain move tables.");
    }

    // Check if any of the input models is a stereo, to report a clear error that this is not supported.
    for (const std::string& model : bam_info.basecaller_models) {
        if (model.find("stereo") != std::string::npos) {
            std::ostringstream oss;
            oss << "Inputs from duplex basecalling are not supported. Detected model: '" << model
                << "' in the input BAM.";
            if (!any_model) {
                throw std::runtime_error{oss.str()};
            } else {
                spdlog::warn("{} This may produce inferior results.", oss.str());
            }
        }
    }

    // Fail only if not explicitly permitting any model, or if any model is allowed but user specified
    // auto model resolution (in which case, the model name needs to be available in the input BAM file).
    if (!any_model && (std::size(bam_info.basecaller_models) != 1)) {
        if (std::empty(bam_info.basecaller_models)) {
            throw std::runtime_error{
                    "Input BAM file has no basecaller models listed in the header."};
        }
        if (std::size(bam_info.basecaller_models) > 1) {
            std::ostringstream oss;
            oss << "Input BAM file has a mix of different basecaller models. Only one basecaller "
                   "model can be processed. List of all basecaller models found in the BAM file:\n";
            print_basecaller_models(oss, bam_info.basecaller_models, ", ");
            throw std::runtime_error{oss.str()};
        }
    }

    std::filesystem::path model_dir;

    if (!std::empty(model_str) && std::filesystem::exists(model_str)) {
        spdlog::debug("Resolved model from user-specified path: {}", model_str);
        spdlog::info("Model specified by path: '{}'", model_str);
        model_dir = model_str;

    } else if (count_model_hits(models::polish_models(), model_str) == 1) {
        const std::string& model_name = model_str;
        spdlog::debug("Resolved model from user-specified polishing model name: {}", model_name);
        spdlog::info("Downloading model: '{}'", model_name);
        model_downloader::ModelDownloader downloader(models_directory);
        model_dir = downloader.get(model_name, "polish");

    } else {
        throw std::runtime_error{"Could not resolve model from string: '" + model_str + "'."};
    }

    return model_dir;
}

void validate_bam_model(const secondary::BamInfo& bam_info,
                        const secondary::ModelConfig& model_config,
                        const bool bacteria,
                        const bool any_model,
                        const secondary::LabelSchemeType expected_label_scheme) {
    // Check that both the model and data have dwells, or that they both do not have dwells.
    const auto it_dwells = model_config.model_kwargs.find("use_dwells");
    const bool model_uses_dwells = (it_dwells != std::end(model_config.model_kwargs))
                                           ? (it_dwells->second == "true")
                                           : false;

    const bool run_dwell_check =
            !bacteria && (legacy_basecaller_models.count(model_config.basecaller_model) == 0);

    const bool label_scheme_is_compatible =
            secondary::parse_label_scheme_type(model_config.label_scheme_type) ==
            expected_label_scheme;

    const auto check_models_supported =
            [bacteria, &model_config](const std::unordered_set<std::string>& basecaller_models) {
                // Every model from the input BAM needs to be supported by the selected polishing model.
                for (const std::string& model : basecaller_models) {
                    const std::unordered_set<std::string> compatible_models =
                            get_compatible_model_set(model, bacteria);
                    const bool valid =
                            sets_intersect(compatible_models, model_config.supported_basecallers);
                    if (!valid) {
                        return false;
                    }
                }
                return true;
            };

    if (!any_model) {
        // Verify that the basecaller model of the loaded config is compatible with the BAM.
        if (!check_models_supported(bam_info.basecaller_models)) {
            throw std::runtime_error{"Polishing model is not compatible with the input BAM!"};
        }

        // Fail if the dwell information in the model and the data does not match.
        if (run_dwell_check && bam_info.has_dwells && !model_uses_dwells) {
            throw std::runtime_error{
                    "Input data has move tables, but a model without move table support has been "
                    "chosen."};
        } else if (!bam_info.has_dwells && model_uses_dwells) {
            throw std::runtime_error{
                    "Input data does not contain move tables, but a model which requires move "
                    "tables has been chosen."};
        }

        if (!label_scheme_is_compatible) {
            throw std::runtime_error{
                    "Incompatible model label scheme! Expected HaploidLabelScheme but got " +
                    model_config.label_scheme_type + "."};
        }

    } else {
        // Allow to use a polishing model trained on a wrong basecaller model, but emit a warning.
        if (!check_models_supported(bam_info.basecaller_models)) {
            spdlog::warn(
                    "Polishing model is not compatible with the input BAM. This may produce "
                    "inferior results.");
        }

        // Allow to use a mismatched model, but emit a warning.
        if (run_dwell_check && bam_info.has_dwells && !model_uses_dwells) {
            spdlog::warn(
                    "Input data has move tables, but a model without move table support has been "
                    "chosen. This may produce inferior results.");
        } else if (!bam_info.has_dwells && model_uses_dwells) {
            spdlog::warn(
                    "Input data does not contain move tables, but a model which requires move "
                    "tables has been chosen. This may produce inferior results.");
        }

        if (!label_scheme_is_compatible) {
            spdlog::warn("Incompatible model label scheme! Expected HaploidLabelScheme but got " +
                         model_config.label_scheme_type + ". This may produce unexpected results.");
        }
    }
}

void run_polishing(const Options& opt,
                   polisher::PolisherResources& resources,
                   polisher::PolishProgressTracker& tracker,
                   secondary::Stats& stats) {
    spdlog::info("Threads: {}, inference threads: {}, number of devices: {}", opt.threads,
                 opt.infer_threads, std::size(resources.devices));

    at::InferenceMode infer_guard;

    // Create a .fai index if it doesn't exist.
    const bool rv_fai = utils::create_fai_index(opt.in_draft_fastx_fn);
    if (!rv_fai) {
        spdlog::error("Failed to create/verify a .fai index for input file: '{}'!",
                      opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    // Load sequence lengths.
    spdlog::debug("[run_polishing] Loading draft sequence lengths.");
    const std::vector<std::pair<std::string, int64_t>> draft_lens =
            utils::load_seq_lengths(opt.in_draft_fastx_fn);

    // Create windows only for the selected regions.
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> draft_lookup;
    for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
        draft_lookup[draft_lens[seq_id].first] = {seq_id, draft_lens[seq_id].second};
    }

    secondary::validate_regions(opt.regions, draft_lens);

    // Open the draft FASTA file. One reader per thread.
    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> draft_readers;
    draft_readers.reserve(opt.threads);
    for (int32_t i = 0; i < opt.threads; ++i) {
        draft_readers.emplace_back(
                std::make_unique<hts_io::FastxRandomReader>(opt.in_draft_fastx_fn));
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

    // Open the output stream to a file/stdout for the consensus sequences.
    const std::string bn =
            (opt.out_format == OutputFormat::FASTA) ? "consensus.fasta" : "consensus.fastq";
    const std::filesystem::path out_consensus_fn =
            (std::empty(opt.output_dir)) ? "" : (opt.output_dir / bn);
    auto ofs_consensus = utils::get_output_stream(out_consensus_fn);

    // Open the output stream to a file/stdout for the variant calls.
    const std::filesystem::path out_vcf_fn =
            (std::empty(opt.output_dir)) ? "-" : (opt.output_dir / "variants.vcf");

    // VCF writer, nullptr unless variant calling is run.
    std::unique_ptr<secondary::VCFWriter> vcf_writer;

    if (opt.run_variant_calling) {
        // These are the only available FILTER options.
        const std::vector<std::pair<std::string, std::string>> filters{
                {"PASS", "All filters passed"},
                {"LowQual", "Variant quality is below threshold"},
                {".", "Non-variant position"},
        };

        vcf_writer = std::make_unique<secondary::VCFWriter>(out_vcf_fn, filters, draft_lens);
    }

    // Prepare regions for processing.
    const auto [input_regions, region_batches] =
            secondary::prepare_region_batches(draft_lens, opt.regions, opt.draft_batch_size);

    // Update the progress tracker.
    {
        int64_t total_input_bases =
                std::accumulate(std::begin(input_regions), std::end(input_regions),
                                static_cast<int64_t>(0), [](const int64_t a, const auto& b) {
                                    int64_t sum = 0;
                                    for (const auto& region : b) {
                                        sum += region.end - region.start;
                                    }
                                    return a + sum;
                                });

        // Variant calling likely takes much less time than consensus,
        // but we need an estimate.
        if (opt.run_variant_calling) {
            total_input_bases *= 2;
        }

        stats.set("total", static_cast<double>(total_input_bases));
        stats.set("processed", 0.0);
    }

    // Compute the minimum usable memory across all devices and use that as the batch size.
    // Reason: batches are constructed and pushed to a queue, workers only pop the batches from
    // the queue, and minimum possible batch size needs to be satisfied.
    const double min_avail_mem = [&resources]() {
        if (std::empty(resources.devices)) {
            return 0.0;
        }
        double ret = resources.devices.front().available_memory_GB;
        for (const polisher::DeviceInfo& device_info : resources.devices) {
            ret = std::min(ret, device_info.available_memory_GB);
        }
        return ret;
    }();
    constexpr double AVAILABLE_MEMORY_FACTOR = 0.85;
    const double usable_mem = (min_avail_mem * AVAILABLE_MEMORY_FACTOR) / opt.infer_threads;

    if (opt.batch_size > 0) {
        spdlog::info("Using fixed batch size: {}", opt.batch_size);
    } else {
        spdlog::info("Using auto computed batch size. Usable per-worker memory: {:.2f} GB",
                     usable_mem);
    }

    int64_t total_batch_bases = 0;
    std::atomic<bool> worker_terminate{false};

    // Process the draft sequences in batches of user-specified size.
    for (const auto& batch_interval : region_batches) {
        // Get the regions for this interval.
        std::vector<secondary::Region> region_batch;
        for (int32_t i = batch_interval.start; i < batch_interval.end; ++i) {
            region_batch.insert(std::end(region_batch), std::begin(input_regions[i]),
                                std::end(input_regions[i]));
        }

        // Total number of bases in this batch.
        const int64_t batch_bases = std::accumulate(
                std::begin(region_batch), std::end(region_batch), static_cast<int64_t>(0),
                [](const int64_t a, const auto& b) { return a + b.end - b.start; });

        // Debug print.
        spdlog::debug("[run_polishing] =============================");
        spdlog::debug("[run_polishing] Processing batch interval of drafts: [{}, {})",
                      batch_interval.start, batch_interval.end);
        for (int64_t i = 0; i < dorado::ssize(region_batch); ++i) {
            spdlog::debug("[run_polishing] region_batch i = {}: {}", i,
                          secondary::region_to_string(region_batch[i]));
        }

        std::vector<std::vector<secondary::ConsensusResult>> all_results_cons;
        std::vector<secondary::VariantCallingSample> vc_input_data;

        // Inference and consensus.
        try {
            // Profiling block.
            {
                utils::ScopedProfileRange spr1("run-prep_infer_decode", 1);

                // Split the sequences into larger BAM windows, like Medaka.
                // NOTE: the window.seq_id is the _absolute_ sequence ID of the input draft sequences.
                spdlog::debug("Creating BAM windows.");
                const std::vector<secondary::Window> bam_regions =
                        polisher::create_windows_from_regions(region_batch, draft_lookup,
                                                              opt.bam_chunk, opt.window_overlap);

                spdlog::debug(
                        "[run_polishing] Starting to produce consensus for regions: {}-{}/{} "
                        "(number: {}, total "
                        "length: {:.2f} Mbp)",
                        batch_interval.start, batch_interval.end, std::size(input_regions),
                        std::size(region_batch), batch_bases / (1000.0 * 1000.0));

                // Update the tracker title.
                {
                    std::ostringstream oss;
                    oss << batch_interval.start << "-" << batch_interval.end << "/"
                        << std::size(input_regions) << ", bases: " << batch_bases;
                    tracker.set_description("Polishing draft sequences: " + oss.str());
                }

                // Each item is one batch for inference.
                utils::AsyncQueue<polisher::InferenceData> batch_queue(opt.queue_size);
                utils::AsyncQueue<polisher::DecodeData> decode_queue(opt.queue_size);

                // Create a thread for the sample producer.
                polisher::WorkerReturnStatus wrs_sample_producer;
                auto thread_sample_producer =
                        utils::jthread([&resources, &bam_regions, &draft_lens, &opt, &usable_mem,
                                        &batch_queue, &worker_terminate, &wrs_sample_producer] {
                            utils::set_thread_name("polish_produce");
                            polisher::sample_producer(
                                    resources, bam_regions, draft_lens, opt.threads, opt.batch_size,
                                    opt.window_len, opt.window_overlap, opt.bam_subchunk,
                                    usable_mem, opt.continue_on_error, batch_queue,
                                    worker_terminate, wrs_sample_producer);
                        });

                // Create a thread for the sample decoder.
                polisher::WorkerReturnStatus wrs_decoder;
                auto thread_sample_decoder =
                        utils::jthread([&all_results_cons, &vc_input_data, &decode_queue, &stats,
                                        &resources, &opt, &worker_terminate, &wrs_decoder] {
                            utils::set_thread_name("polish_decode");
                            polisher::decode_samples_in_parallel(
                                    all_results_cons, vc_input_data, decode_queue, stats,
                                    worker_terminate, wrs_decoder, *resources.decoder, opt.threads,
                                    opt.min_depth, opt.run_variant_calling, opt.continue_on_error);
                        });

                // Run the inference worker on the main thread.
                polisher::infer_samples_in_parallel(
                        batch_queue, decode_queue, resources.models, worker_terminate,
                        resources.streams, resources.encoders, draft_lens, opt.continue_on_error);

                // Join the workers.
                thread_sample_producer.join();
                thread_sample_decoder.join();

                // Propagate worker errors into the main thread.
                if (wrs_sample_producer.exception_thrown) {
                    throw std::runtime_error{wrs_sample_producer.message};
                }
                if (wrs_decoder.exception_thrown) {
                    throw std::runtime_error{wrs_decoder.message};
                }
            }

        } catch (const std::exception& e) {
            if (!opt.continue_on_error) {
                throw;
            } else {
                spdlog::warn(
                        "Exception caught when running inference on the batch interval of drafts: "
                        "[{}, {}). Skipping this batch and optionally outputting unpolished "
                        "sequences. Original exception: \"{}\"",
                        batch_interval.start, batch_interval.end, e.what());
            }
        }

        // Write the consensus. If a sequence has no inferred samples, it can be
        // written verbatim to the output.
        // If this fails, stop execution.
        try {
            utils::ScopedProfileRange spr1("run-construct_consensus_and_write", 1);

            // Round the counter, in case some samples were dropped.
            total_batch_bases += batch_bases;
            stats.set("processed", static_cast<double>(total_batch_bases));

            spdlog::debug(
                    "[run_polishing] Stitching sequences: {}-{}/{} (number: {}, total "
                    "length: {:.2f} Mbp), parts: {}",
                    batch_interval.start, batch_interval.end, std::size(input_regions),
                    std::size(region_batch), batch_bases / (1000.0 * 1000.0),
                    std::size(all_results_cons));

            spdlog::debug("Data for variant calling: num elements = {}, num consensus results = {}",
                          std::size(vc_input_data), std::size(all_results_cons));

            // Construct the consensus sequences, only if they will be written.
            if (opt.write_consensus) {
                utils::ScopedProfileRange spr2("run-construct_seqs_and_write", 2);

                // Dimensions: [draft_id x part_id x haplotype_id].
                const std::vector<std::vector<std::vector<secondary::ConsensusResult>>>
                        consensus_seqs = polisher::construct_consensus_seqs(
                                batch_interval, all_results_cons, draft_lens, opt.fill_gaps,
                                opt.fill_char, *draft_readers.front());

                // Write the consensus file.
                for (const auto& consensus : consensus_seqs) {
                    write_consensus_results(*ofs_consensus, consensus, opt.fill_gaps,
                                            (opt.out_format == OutputFormat::FASTQ));
                }
            }
        } catch (const std::exception& e) {
            if (!opt.continue_on_error) {
                throw;
            } else {
                spdlog::warn(
                        "Exception caught when writing consensus sequences on interval of drafts: "
                        "[{}, {}). Skipping this batch. Original exception: \"{}\"",
                        batch_interval.start, batch_interval.end, e.what());
                continue;
            }
        }

        // Variant calling.
        try {
            utils::ScopedProfileRange spr1("run-variant_calling", 1);

            // Run variant calling, optionally.
            if (opt.run_variant_calling) {
                std::vector<secondary::Variant> variants = polisher::call_variants(
                        worker_terminate, stats, batch_interval, vc_input_data, draft_readers,
                        draft_lens, *resources.decoder, opt.ambig_ref,
                        opt.vc_type == VariantCallingEnum::GVCF, opt.threads,
                        opt.continue_on_error);

                std::sort(std::begin(variants), std::end(variants),
                          [](const auto& a, const auto& b) {
                              return std::tie(a.seq_id, a.pos) < std::tie(b.seq_id, b.pos);
                          });

                // Write the VCF file.
                for (const auto& variant : variants) {
                    vcf_writer->write_variant(variant);
                }

                // We approximate the progress by expecting 2x bases to be processed
                // when doing variant calling.
                total_batch_bases += batch_bases;
                stats.set("processed", static_cast<double>(total_batch_bases));
            }
        } catch (const std::exception& e) {
            if (!opt.continue_on_error) {
                throw;
            } else {
                spdlog::warn(
                        "Exception caught when calling variants in the batch interval of drafts: "
                        "[{}, {}). Not producing variant calls for this batch of drafts. Original "
                        "exception: \"{}\"",
                        batch_interval.start, batch_interval.end, e.what());
            }
        }
    }
}

}  // namespace

int polish(int argc, char* argv[]) {
    try {
        // Initialize CLI options. The parse_args below requires a non-const reference.
        // Verbosity is passed into a callback, so we need it here.
        int verbosity = 0;
        argparse::ArgumentParser parser("dorado polish", DORADO_VERSION,
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
            spdlog::debug("bam_info.uses_dorado_aligner = {}", bam_info.uses_dorado_aligner);
            spdlog::debug("bam_info.has_dwells = {}", bam_info.has_dwells);
            spdlog::debug("bam_info.read_groups:");
            for (const auto& rg : bam_info.read_groups) {
                spdlog::debug("    - {}", rg);
            }
            spdlog::debug("bam_info.basecaller_models:");
            for (const auto& rg : bam_info.basecaller_models) {
                spdlog::debug("    - {}", rg);
            }
            if (!std::empty(opt.read_group)) {
                spdlog::debug(
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
            // Basic mainstream model resolving.
            const std::filesystem::path model_dir =
                    resolve_model(bam_info, opt.models_directory, opt.bacteria);
            model_config = polisher::load_model(model_dir, opt.load_scripted_model);
            validate_bam_model(bam_info, model_config, opt.bacteria, false,
                               secondary::LabelSchemeType::HAPLOID);
        } else {
            // Advanced model resolve from a specific path or model name.
            const std::filesystem::path model_dir = resolve_model_advanced(
                    bam_info, opt.models_directory, opt.model_str, opt.any_model);
            model_config = polisher::load_model(model_dir, opt.load_scripted_model);
            validate_bam_model(bam_info, model_config, opt.bacteria, opt.any_model,
                               secondary::LabelSchemeType::HAPLOID);
        }

        // Create the models, encoders and BAM handles.
        polisher::PolisherResources resources = polisher::create_resources(
                model_config, opt.in_draft_fastx_fn, opt.in_aln_bam_fn, opt.device_str, opt.threads,
                opt.infer_threads, opt.full_precision, opt.read_group, opt.tag_name, opt.tag_value,
                opt.tag_keep_missing, opt.min_mapq, std::nullopt, std::nullopt);

        // Progress bar.
        secondary::Stats stats;
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        polisher::PolishProgressTracker tracker;
        std::vector<dorado::stats::StatsCallable> stats_callables;
        stats_callables.push_back([&tracker, &stats](const stats::NamedStats& /*stats*/) {
            tracker.update_progress_bar(stats.get_stats());
        });
        constexpr auto kStatsPeriod = std::chrono::milliseconds(1000);
        auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));

        run_polishing(opt, resources, tracker, stats);

        tracker.finalize();
        stats_sampler->terminate();

        // Hack to clear the last line from the progress bar. The library automatically does '\r'.
        std::cerr << std::string(200, ' ') << '\r';
        spdlog::info("Done!");

    } catch (const std::exception& e) {
        spdlog::error(e.what());
        return EXIT_FAILURE;
    } catch (...) {
        spdlog::error("Caught an unknown exception!");
        return EXIT_FAILURE;
    }

    return 0;
}

}  // namespace dorado
