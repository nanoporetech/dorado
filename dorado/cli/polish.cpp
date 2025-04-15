#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "hts_io/FastxRandomReader.h"
#include "model_downloader/model_downloader.h"
#include "models/models.h"
#include "polish/polish_impl.h"
#include "polish/polish_progress_tracker.h"
#include "secondary/architectures/model_config.h"
#include "secondary/bam_info.h"
#include "secondary/batching.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/vcf_writer.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/torch_utils.h"
#include "utils/AsyncQueue.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/io_utils.h"
#include "utils/log_utils.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"

#include <ATen/Parallel.h>
#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

#include <algorithm>
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

using ParserPtr = std::unique_ptr<utils::arg_parse::ArgParser>;

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
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado polish");

    parser->visible.add_description("Consensus tool for polishing draft assemblies");

    {
        // Positional arguments group
        parser->visible.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser->visible.add_argument("in_draft_fastx").help("Draft assembly for polishing");
    }
    {
        // Default "Optional arguments" group
        parser->visible.add_argument("-t", "--threads")
                .help("Number of threads for processing (0=unlimited).")
                .default_value(0)
                .scan<'i', int>();

        parser->visible.add_argument("--infer-threads")
                .help("Number of threads for inference")
#if DORADO_CUDA_BUILD
                .default_value(2)
#else
                .default_value(1)
#endif
                .scan<'i', int>();

        parser->visible.add_argument("-x", "--device")
                .help(std::string{"Specify CPU or GPU device: 'auto', 'cpu', 'cuda:all' or "
                                  "'cuda:<device_id>[,<device_id>...]'. Specifying 'auto' will "
                                  "choose either 'cpu' "
                                  "or 'cuda:all' depending on the presence of a GPU device."})
                .default_value(std::string{"auto"});

        parser->visible.add_argument("-v", "--verbose")
                .flag()
                .action([&](const auto&) { ++verbosity; })
                .append();
    }
    {
        parser->visible.add_group("Input/output options");
        parser->visible.add_argument("-o", "--output-dir")
                .help("If specified, output files will be written to the given folder. Otherwise, "
                      "output is to stdout.")
                .default_value("");
        parser->visible.add_argument("-m", "--model")
                .help("Path to correction model folder.")
                .default_value("auto");
        parser->visible.add_argument("--bacteria")
                .help("Optimise polishing for plasmids and bacterial genomes.")
                .flag();
        parser->visible.add_argument("-q", "--qualities")
                .help("Output with per-base quality scores (FASTQ).")
                .flag();
        parser->visible.add_argument("--vcf")
                .help("Output a VCF file with variant calls to --output-dir if specified, "
                      "otherwise to stdout.")
                .flag();
        parser->visible.add_argument("--gvcf")
                .help("Output a gVCF file to --output-dir if specified, otherwise to stdout.")
                .flag();
        parser->visible.add_argument("--ambig-ref")
                .help("Decode variants at ambiguous reference positions.")
                .flag();
    }
    {
        parser->visible.add_group("Advanced options");
        parser->visible.add_argument("-b", "--batchsize")
                .help("Batch size for inference.")
                .default_value(16)
                .scan<'i', int>();
        parser->visible.add_argument("--draft-batchsize")
                .help("Approximate batch size for processing input draft sequences.")
                .default_value(std::string{"200M"});
        parser->visible.add_argument("--window-len")
                .help("Window size for calling consensus.")
                .default_value(10000)
                .scan<'i', int>();
        parser->visible.add_argument("--window-overlap")
                .help("Overlap length between windows.")
                .default_value(1000)
                .scan<'i', int>();
        parser->visible.add_argument("--bam-chunk")
                .help("Size of draft chunks to parse from the input BAM at a time.")
                .default_value(1000000)
                .scan<'i', int>();
        parser->visible.add_argument("--bam-subchunk")
                .help("Size of regions to split the bam_chunk in to for parallel processing")
                .default_value(100000)
                .scan<'i', int>();
        parser->visible.add_argument("--no-fill-gaps")
                .help("Do not fill gaps in consensus sequence with draft sequence.")
                .flag();
        parser->visible.add_argument("--fill-char")
                .help("Use a designated character to fill gaps.");
        parser->visible.add_argument("--regions")
                .help("Process only these regions of the input. Can be either a path to a BED file "
                      "or a list of comma-separated Htslib-formatted regions (start is 1-based, "
                      "end "
                      "is inclusive).");
        parser->visible.add_argument("--RG").help("Read group to select.").default_value("");
        parser->visible.add_argument("--ignore-read-groups")
                .help("Ignore read groups in bam file.")
                .flag();
        parser->visible.add_argument("--tag-name")
                .help("Two-letter BAM tag name for filtering the alignments during feature "
                      "generation")
                .default_value("");
        parser->visible.add_argument("--tag-value")
                .help("Value of the tag for filtering the alignments during feature generation")
                .default_value(0)
                .scan<'i', int>();

        parser->visible.add_argument("--tag-keep-missing")
                .help("Keep alignments when tag is missing. If specified, overrides "
                      "the same option in the model config.")
                .flag();
        parser->visible.add_argument("--min-mapq")
                .help("Minimum mapping quality of the input alignments. If specified, overrides "
                      "the same option in the model config.")
                .scan<'i', int>();
        parser->visible.add_argument("--min-depth")
                .help("Sites with depth lower than this value will not be polished.")
                .default_value(0)
                .scan<'i', int>();
    }

    // Hidden advanced arguments.
    {
        parser->hidden.add_argument("--full-precision")
                .help("Always use full precision for inference.")
                .flag();
        parser->hidden.add_argument("--queue-size")
                .help("Queue size for processing.")
                .default_value(1000)
                .scan<'i', int>();
        parser->hidden.add_argument("--scripted")
                .help("Load the scripted Torch model instead of building one internally.")
                .flag();
        parser->hidden.add_argument("--any-bam")
                .help("Allow any BAM as input, not just Dorado aligned.")
                .flag();
        parser->hidden.add_argument("--skip-model-compatibility-check")
                .help("Allow any model to be applied on the data.")
                .flag();
    }

    return parser;
}

int parse_args(int argc, char** argv, utils::arg_parse::ArgParser& parser) {
    try {
        utils::arg_parse::parse(parser, argc, argv);

    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/// \brief This function simply fills out the Options struct with the parsed CLI args.
Options set_options(const utils::arg_parse::ArgParser& parser, const int verbosity) {
    Options opt;

    opt.in_aln_bam_fn = parser.visible.get<std::string>("in_aln_bam");
    opt.in_draft_fastx_fn = parser.visible.get<std::string>("in_draft_fastx");

    opt.output_dir = parser.visible.get<std::string>("output-dir");
    opt.model_str = parser.visible.get<std::string>("model");
    opt.bacteria = parser.visible.get<bool>("bacteria");

    opt.out_format =
            parser.visible.get<bool>("qualities") ? OutputFormat::FASTQ : OutputFormat::FASTA;
    opt.threads = parser.visible.get<int>("threads");
    opt.threads = (opt.threads == 0) ? std::thread::hardware_concurrency() : (opt.threads);

    opt.infer_threads = parser.visible.get<int>("infer-threads");

    opt.device_str = parser.visible.get<std::string>("device");

    if (opt.device_str == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        opt.device_str = "cpu";
#else
        opt.device_str = utils::get_auto_detected_device();
#endif
    }

    opt.batch_size = parser.visible.get<int>("batchsize");
    opt.draft_batch_size =
            std::max<int64_t>(0, utils::arg_parse::parse_string_to_size<int64_t>(
                                         parser.visible.get<std::string>("draft-batchsize")));
    opt.window_len = parser.visible.get<int>("window-len");
    opt.window_overlap = parser.visible.get<int>("window-overlap");
    opt.bam_chunk = parser.visible.get<int>("bam-chunk");
    opt.bam_subchunk = parser.visible.get<int>("bam-subchunk");
    opt.verbosity = verbosity;
    opt.regions_str = parser.visible.present<std::string>("regions");
    if (opt.regions_str) {
        opt.regions = secondary::parse_regions(*opt.regions_str);
    }
    opt.min_depth = parser.visible.get<int>("min-depth");

    opt.full_precision = parser.hidden.get<bool>("full-precision");
    opt.load_scripted_model = parser.hidden.get<bool>("scripted");
    opt.queue_size = parser.hidden.get<int>("queue-size");
    opt.any_bam = parser.hidden.get<bool>("any-bam");
    opt.any_model = parser.hidden.get<bool>("skip-model-compatibility-check");

    opt.fill_gaps = !parser.visible.get<bool>("no-fill-gaps");
    opt.fill_char = (parser.visible.is_used("--fill-char"))
                            ? std::optional<char>{parser.visible.get<std::string>("fill-char")[0]}
                            : std::nullopt;
    opt.read_group = (parser.visible.is_used("--RG")) ? parser.visible.get<std::string>("RG") : "";
    opt.ignore_read_groups = parser.visible.get<bool>("ignore-read-groups");
    opt.tag_name = parser.visible.get<std::string>("tag-name");
    opt.tag_value = parser.visible.get<int>("tag-value");
    // The `"--tag-keep-missing` is a special case because it's a flag, and we cannot use `present`.
    opt.tag_keep_missing =
            (parser.visible.is_used("--tag-keep-missing"))
                    ? std::optional<bool>{parser.visible.get<bool>("tag-keep-missing")}
                    : std::nullopt;
    opt.min_mapq = parser.visible.present<int32_t>("min-mapq");

    if (opt.bam_subchunk > opt.bam_chunk) {
        spdlog::warn(
                "BAM sub-chunk size is larger than bam_chunk size. Limiting to bam_chunk size. "
                "bam_subchunk = {}, bam_chunk = {}",
                opt.bam_chunk, opt.bam_subchunk);
        opt.bam_subchunk = opt.bam_chunk;
    }

    // Variant calling setup.
    const bool vcf = parser.visible.get<bool>("vcf");
    const bool gvcf = parser.visible.get<bool>("gvcf");
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
    opt.ambig_ref = parser.visible.get<bool>("ambig-ref");

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
    if (opt.batch_size <= 0) {
        spdlog::error("Batch size should be > 0. Given: {}.", opt.batch_size);
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

void write_consensus_results(std::ostream& os,
                             const std::vector<secondary::ConsensusResult>& results,
                             const bool fill_gaps,
                             const bool write_quals) {
    if (std::empty(results)) {
        return;
    }

    for (size_t i = 0; i < std::size(results); ++i) {
        secondary::ConsensusResult out = results[i];
        polisher::remove_deletions(out);

        std::string header = results[i].name;
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

std::filesystem::path download_model(const std::string& model_name) {
    const std::filesystem::path tmp_dir = utils::get_downloads_path(std::nullopt);
    const bool success = model_downloader::download_models(tmp_dir.string(), model_name);
    if (!success) {
        spdlog::error("Could not download model: {}", model_name);
        std::exit(EXIT_FAILURE);
    }
    return (tmp_dir / model_name);
}

const secondary::ModelConfig resolve_model(const secondary::BamInfo& bam_info,
                                           const std::string& model_str,
                                           const bool load_scripted_model,
                                           const bool bacteria,
                                           const bool any_model) {
    const auto count_model_hits = [](const dorado::models::ModelList& model_list,
                                     const std::string& model_name) {
        int32_t num_found = 0;
        for (const auto& info : model_list) {
            if (info.name == model_name) {
                ++num_found;
            }
        }
        return num_found;
    };

    // clang-format off
    const std::unordered_map<std::string, std::string> compatible_bacterial_bc_models {
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.3.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
        {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
        {"dna_r10.4.1_e8.2_400bps_sup@v5.0.0", "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0"},
    };
    const std::unordered_map<std::string, std::string> legacy_bc_models {
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_polish"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_polish"},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.3.0", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish"},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.3.0", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish"},
    };
    // clang-format on

    const auto determine_model_name = [&bacteria, &compatible_bacterial_bc_models,
                                       &legacy_bc_models,
                                       &bam_info](const std::string& basecaller_model) {
        if (bacteria) {
            // Resolve a bacterial model.
            const auto it = compatible_bacterial_bc_models.find(basecaller_model);
            if (it == std::cend(compatible_bacterial_bc_models)) {
                throw std::runtime_error(
                        "There are no bacterial models compatible with basecaller model: '" +
                        basecaller_model + "'.");
            }
            return it->second;
        } else {
            // First check if this is a legacy model.
            const auto it_legacy = legacy_bc_models.find(basecaller_model);
            if (it_legacy != std::cend(legacy_bc_models)) {
                return it_legacy->second;
            }

            // Otherwise, this is a current model.
            const std::string polish_model_suffix =
                    std::string("_polish_rl") + (bam_info.has_dwells ? "_mv" : "");
            return basecaller_model + polish_model_suffix;
        }
    };
    const auto sets_intersect = [](const std::unordered_set<std::string>& set1,
                                   const std::unordered_set<std::string>& set2) {
        for (const auto& val : set1) {
            if (set2.count(val) > 0) {
                return true;
            }
        }
        return false;
    };

    std::filesystem::path model_dir;

    if (bam_info.has_dwells) {
        spdlog::info("Input data contains move tables.");
    } else {
        spdlog::info("Input data does not contain move tables.");
    }

    // Fail only if not explicitly permitting any model, or if any model is allowed but user specified
    // auto model resolution (in which case, the model name needs to be available in the input BAM file).
    if (!any_model ||
        (any_model && (model_str == "auto") && (std::size(bam_info.basecaller_models) != 1))) {
        const std::string suffix{(any_model) ? " Cannot use 'auto' to resolve the model." : ""};
        if (std::empty(bam_info.basecaller_models)) {
            throw std::runtime_error{
                    "Input BAM file has no basecaller models listed in the header." + suffix};
        }
        if (std::size(bam_info.basecaller_models) > 1) {
            throw std::runtime_error{
                    "Input BAM file has a mix of different basecaller models. Only one basecaller "
                    "model can be processed." +
                    suffix};
        }
    }

    if (model_str == "auto") {
        spdlog::info("Auto resolving the model.");

        // Check that there is at least one basecaller listed in the BAM. Otherwise, no auto resolving.
        if (std::empty(bam_info.basecaller_models)) {
            throw std::runtime_error{
                    "Cannot auto resolve the model because no model information is available "
                    "in the BAM file."};
        }

        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0
        const std::string& basecaller_model = *std::begin(bam_info.basecaller_models);

        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv
        const std::string model_name = determine_model_name(basecaller_model);

        spdlog::debug("Resolved model from input data: {}", model_name);

        spdlog::info("Downloading model: '{}'", model_name);
        model_dir = download_model(model_name);

    } else if (!std::empty(model_str) && std::filesystem::exists(model_str)) {
        spdlog::debug("Resolved model from user-specified path: {}", model_str);
        spdlog::info("Model specified by path: '{}'", model_str);
        model_dir = model_str;

    } else if (count_model_hits(models::polish_models(), model_str) == 1) {
        const std::string& model_name = model_str;
        spdlog::debug("Resolved model from user-specified polishing model name: {}", model_name);
        spdlog::info("Downloading model: '{}'", model_name);
        model_dir = download_model(model_name);

    } else if (count_model_hits(models::simplex_models(), model_str) == 1) {
        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0
        const std::string& basecaller_model = model_str;

        // Example: dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv
        const std::string model_name = determine_model_name(basecaller_model);

        spdlog::debug("Resolved model from user-specified basecaller model name: {}", model_name);
        spdlog::info("Downloading model: '{}'", model_name);
        model_dir = download_model(model_name);

    } else {
        throw std::runtime_error{"Could not resolve model from string: '" + model_str + "'."};
    }

    // Load the model.
    spdlog::info("Parsing the model config: {}", (model_dir / "config.toml").string());
    const std::string model_file = load_scripted_model ? "model.pt" : "weights.pt";
    secondary::ModelConfig model_config =
            secondary::parse_model_config(model_dir / "config.toml", model_file);

    // Check that both the model and data have dwells, or that they both do not have dwells.
    const auto it_dwells = model_config.model_kwargs.find("use_dwells");
    const bool model_uses_dwells = (it_dwells != std::end(model_config.model_kwargs))
                                           ? (it_dwells->second == "true")
                                           : false;

    const bool run_dwell_check =
            !bacteria && (legacy_bc_models.count(model_config.basecaller_model) == 0);

    if (!any_model) {
        // Verify that the basecaller model of the loaded config is compatible with the BAM.
        if (!sets_intersect(bam_info.basecaller_models, model_config.supported_basecallers)) {
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

    } else {
        // Allow to use a polishing model trained on a wrong basecaller model, but emit a warning.
        if (!sets_intersect(bam_info.basecaller_models, model_config.supported_basecallers)) {
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
    }

    return model_config;
}

void run_polishing(const Options& opt,
                   polisher::PolisherResources& resources,
                   polisher::PolishProgressTracker& tracker,
                   polisher::PolishStats& polish_stats) {
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

        polish_stats.set("total", static_cast<double>(total_input_bases));
        polish_stats.set("processed", 0.0);
    }

    int64_t total_batch_bases = 0;

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

        std::vector<secondary::ConsensusResult> all_results_cons;
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

                std::thread thread_sample_producer = std::thread(
                        &polisher::sample_producer, std::ref(resources), std::cref(bam_regions),
                        std::cref(draft_lens), opt.threads, opt.batch_size, opt.window_len,
                        opt.window_overlap, opt.bam_subchunk, std::ref(batch_queue));

                std::thread thread_sample_decoder = std::thread(
                        &polisher::decode_samples_in_parallel, std::ref(all_results_cons),
                        std::ref(vc_input_data), std::ref(decode_queue), std::ref(polish_stats),
                        std::cref(*resources.decoder), opt.threads, opt.min_depth,
                        opt.run_variant_calling);

                polisher::infer_samples_in_parallel(batch_queue, decode_queue, resources.models,
                                                    resources.streams, *resources.encoder);

                if (thread_sample_producer.joinable()) {
                    thread_sample_producer.join();
                }
                if (thread_sample_decoder.joinable()) {
                    thread_sample_decoder.join();
                }
            }

        } catch (const std::exception& e) {
            // Emit a warning, but do not continue the loop, so that the empty sequences are written.
            spdlog::warn(
                    "Exception caught when polishing the batch interval of drafts: [{}, {}). "
                    "Skipping this batch and optionally outputting unpolished sequences. Original "
                    "exception: \"{}\"",
                    batch_interval.start, batch_interval.end, e.what());
        }

        // Write the consensus. If a sequence has no inferred samples, it can be
        // written verbatim to the output.
        // If this fails, stop execution.
        {
            utils::ScopedProfileRange spr1("run-construct_consensus_and_write", 1);

            // Round the counter, in case some samples were dropped.
            total_batch_bases += batch_bases;
            polish_stats.set("processed", static_cast<double>(total_batch_bases));

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

                const std::vector<std::vector<secondary::ConsensusResult>> consensus_seqs =
                        polisher::construct_consensus_seqs(batch_interval, all_results_cons,
                                                           draft_lens, opt.fill_gaps, opt.fill_char,
                                                           *draft_readers.front());

                // Write the consensus file.
                for (const auto& consensus : consensus_seqs) {
                    write_consensus_results(*ofs_consensus, consensus, opt.fill_gaps,
                                            (opt.out_format == OutputFormat::FASTQ));
                }
            }
        }

        // Variant calling.
        try {
            utils::ScopedProfileRange spr1("run-variant_calling", 1);

            // Run variant calling, optionally.
            if (opt.run_variant_calling) {
                std::vector<secondary::Variant> variants = call_variants(
                        batch_interval, vc_input_data, draft_readers, draft_lens,
                        *resources.decoder, opt.ambig_ref, opt.vc_type == VariantCallingEnum::GVCF,
                        opt.threads, polish_stats);

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
                polish_stats.set("processed", static_cast<double>(total_batch_bases));
            }
        } catch (const std::exception& e) {
            spdlog::warn(
                    "Exception caught when calling variants in the batch interval of drafts: [{}, "
                    "{}). Not producing variant calls for this batch of drafts. Original "
                    "exception: \"{}\"",
                    batch_interval.start, batch_interval.end, e.what());
            continue;
        }
    }
}

}  // namespace

int polish(int argc, char* argv[]) {
    try {
        // Initialize CLI options. The parse_args below requires a non-const reference.
        // Verbosity is passed into a callback, so we need it here.
        int verbosity = 0;
        ParserPtr parser = create_cli(verbosity);

        // Parse the arguments.
        const int rv_parse = parse_args(argc, argv, *parser);

        if (rv_parse != EXIT_SUCCESS) {
            return rv_parse;
        }

        // Initialize the options from the CLI.
        const Options opt = set_options(*parser, verbosity);

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

        // Resolve the model for polishing.
        const secondary::ModelConfig model_config = resolve_model(
                bam_info, opt.model_str, opt.load_scripted_model, opt.bacteria, opt.any_model);

        // Create the models, encoders and BAM handles.
        polisher::PolisherResources resources = polisher::create_resources(
                model_config, opt.in_aln_bam_fn, opt.device_str, opt.threads, opt.infer_threads,
                opt.full_precision, opt.read_group, opt.tag_name, opt.tag_value,
                opt.tag_keep_missing, opt.min_mapq);

        // Progress bar.
        polisher::PolishStats polish_stats;
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        polisher::PolishProgressTracker tracker;
        std::vector<dorado::stats::StatsCallable> stats_callables;
        stats_callables.push_back([&tracker, &polish_stats](const stats::NamedStats& /*stats*/) {
            tracker.update_progress_bar(polish_stats.get_stats());
        });
        constexpr auto kStatsPeriod = std::chrono::milliseconds(1000);
        auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));

        run_polishing(opt, resources, tracker, polish_stats);

        tracker.finalize();
        stats_sampler->terminate();

        // Hack to clear the last line from the progress bar. The library automatically does '\r'.
        std::cerr << std::string(200, ' ') << '\r';
        spdlog::info("Done!");

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        spdlog::error("Caught an unknown exception!");
        return EXIT_FAILURE;
    }

    return 0;
}

}  // namespace dorado
