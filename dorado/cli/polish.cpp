#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "hts_io/FastxRandomReader.h"
#include "model_downloader/model_downloader.h"
#include "models/models.h"
#include "polish/architectures/model_config.h"
#include "polish/interval.h"
#include "polish/polish_impl.h"
#include "polish/polish_progress_tracker.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/AsyncQueue.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"

#include <ATen/Parallel.h>
#include <IntervalTree.h>
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
    bool infer_threads_is_set = false;
    std::string device_str;
    int32_t batch_size = 16;
    int64_t draft_batch_size = 200'000'000;
    int32_t window_len = 10000;
    int32_t window_overlap = 1000;
    int32_t bam_chunk = 1'000'000;
    int32_t bam_subchunk = 100'000;
    std::optional<std::string> regions_str;
    std::vector<polisher::Region> regions;
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

/// \brief Parses Htslib-style regions either from a BED-like file on disk, or from a comma
///         separated list of regions in the given string.
std::vector<polisher::Region> parse_regions(const std::string& regions_arg) {
    if (std::empty(regions_arg)) {
        return {};
    }

    std::vector<polisher::Region> ret;

    // Check if the string points to a file on disk.
    if (std::filesystem::exists(regions_arg)) {
        // Parse the BED-like format.
        std::ifstream ifs(regions_arg);
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string chr;
            int64_t start = 0;
            int64_t end = 0;
            iss >> chr >> start >> end;
            ret.emplace_back(polisher::Region{chr, start, end});
        }

    } else {
        // Parse a comma delimited string of regions.
        const auto str_regions = dorado::utils::split(regions_arg, ',');
        for (const std::string& str_region : str_regions) {
            polisher::Region region = polisher::parse_region_string(str_region);
            ret.emplace_back(region);
        }
    }

    return ret;
}

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
                .help("Number of threads for CPU inference")
                .default_value(1)
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
                .help("Output a VCF file with variant calls to stdout.")
                .flag();
        parser->visible.add_argument("--gvcf").help("Output a gVCF file to stdout.").flag();
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
    opt.infer_threads_is_set = parser.visible.is_used("--infer-threads");

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
        opt.regions = parse_regions(*opt.regions_str);
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
    if (!std::empty(opt.output_dir) || vcf || gvcf) {
        opt.run_variant_calling = true;
    }
    if (vcf && gvcf) {
        spdlog::warn("Both --vcf and --gvcf are specified. gVCF will be output.");
        opt.vc_type = VariantCallingEnum::GVCF;
    } else if (vcf) {
        opt.vc_type = VariantCallingEnum::VCF;
    } else if (gvcf) {
        opt.vc_type = VariantCallingEnum::GVCF;
    }
    opt.ambig_ref = parser.visible.get<bool>("ambig-ref");

    // Write the consensus sequence only if: (1) to a folder, or (2) to stdout but no VC options were specified.
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

    if ((opt.device_str != "cpu") && opt.infer_threads_is_set) {
        spdlog::error(
                "Specifying the number of CPU inference threads is only allowed when the device is "
                "set to 'cpu'.");
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

std::vector<std::pair<std::string, int64_t>> load_seq_lengths(
        const std::filesystem::path& in_fastx_fn) {
    const std::filesystem::path fai_path = utils::get_fai_path(in_fastx_fn);

    std::vector<std::pair<std::string, int64_t>> ret;
    std::string line;
    std::ifstream ifs(fai_path);
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            continue;
        }
        std::string name;
        int64_t length = 0;
        std::istringstream iss(line);
        iss >> name >> length;
        ret.emplace_back(std::move(name), length);
    }
    return ret;
}

std::vector<std::vector<polisher::ConsensusResult>> construct_consensus_seqs(
        const dorado::polisher::Interval& region_batch,
        const std::vector<polisher::ConsensusResult>& all_results_cons,
        const hts_io::FastxRandomReader& draft_reader,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool fill_gaps,
        const std::optional<char>& fill_char) {
    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(all_results_cons); ++i) {
        const polisher::ConsensusResult& r = all_results_cons[i];
        const int32_t local_id = r.draft_id - region_batch.start;
        // Skip filtered samples.
        if (r.draft_id < 0) {
            continue;
        }
        if ((r.draft_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    r.draft_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(r.draft_start, i);
    }

    std::vector<std::vector<polisher::ConsensusResult>> ret;

    // Consensus sequence - stitch the windows and write output.
    for (int64_t group_id = 0; group_id < dorado::ssize(groups); ++group_id) {
        const int64_t seq_id = group_id + region_batch.start;

        auto& group = groups[group_id];
        std::sort(std::begin(group), std::end(group));  // Sort by start pos.

        const std::string& header = draft_lens[seq_id].first;

        std::vector<polisher::ConsensusResult> consensus = polisher::stitch_sequence(
                draft_reader, header, all_results_cons, group, fill_gaps, fill_char);

        ret.emplace_back(std::move(consensus));
    }

    return ret;
}

void write_consensus_results(std::ostream& os,
                             const std::vector<polisher::ConsensusResult>& results,
                             const bool fill_gaps,
                             const bool write_quals) {
    if (std::empty(results)) {
        return;
    }

    for (size_t i = 0; i < std::size(results); ++i) {
        polisher::ConsensusResult out = results[i];
        remove_deletions(out);

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

std::unique_ptr<std::ostream, void (*)(std::ostream*)> get_output_stream(
        const std::filesystem::path& out_fn) {
    if (std::empty(out_fn)) {
        return {&std::cout, [](std::ostream*) {}};
    }
    std::unique_ptr<std::ofstream, void (*)(std::ostream*)> ofs(
            new std::ofstream(out_fn), [](std::ostream* ptr) { delete ptr; });
    if (!ofs->is_open()) {
        throw std::runtime_error("Failed to open file: " + out_fn.string());
    }
    return ofs;
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

const polisher::ModelConfig resolve_model(const polisher::BamInfo& bam_info,
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
    polisher::ModelConfig model_config =
            polisher::parse_model_config(model_dir / "config.toml", model_file);

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

void check_read_groups(const polisher::BamInfo& bam_info, const std::string& cli_read_group) {
    if (!std::empty(cli_read_group) && std::empty(bam_info.read_groups)) {
        throw std::runtime_error{
                "No @RG headers found in the input BAM, but user-specified RG was given. RG: '" +
                cli_read_group + "'"};

    } else if (std::empty(cli_read_group) && std::size(bam_info.read_groups) > 1) {
        throw std::runtime_error{
                "The input BAM contains more than one read group. Please specify --RG to select "
                "which read group to process."};

    } else if (!std::empty(cli_read_group) && !std::empty(bam_info.read_groups)) {
        if (bam_info.read_groups.count(cli_read_group) == 0) {
            std::ostringstream oss;
            polisher::print_container(oss, bam_info.read_groups, ", ");
            throw std::runtime_error{"Requested RG is not in the input BAM. Requested: '" +
                                     cli_read_group + "'"};
        }
    }
}

/**
 * \brief Checks if any region overlaps amy of the other regions. This may produce wrong results, so
 *          we disallow that.
 */
void validate_regions(const std::vector<polisher::Region>& regions,
                      const std::vector<std::pair<std::string, int64_t>>& seq_lens) {
    // Create intervals for each input sequence.
    std::unordered_map<std::string, std::vector<interval_tree::Interval<int64_t, int64_t>>>
            intervals;
    for (const auto& region : regions) {
        // NOTE: interval_tree has an inclusive end coordinate.
        intervals[region.name].emplace_back(
                interval_tree::Interval<int64_t, int64_t>(region.start, region.end - 1, 0));
    }

    // Compute the interval tree.
    std::unordered_map<std::string, interval_tree::IntervalTree<int64_t, int64_t>> trees;
    for (auto& [key, values] : intervals) {
        trees[key] = interval_tree::IntervalTree<int64_t, int64_t>(std::move(values));
    }

    // Validate that none of the regions is overlapping any other region.
    for (const auto& region : regions) {
        std::vector<interval_tree::Interval<int64_t, int64_t>> results =
                trees[region.name].findOverlapping(region.start, region.end - 1);
        if (std::size(results) > 1) {
            throw std::runtime_error("Region validation failed: region '" +
                                     polisher::region_to_string(region) +
                                     "' overlaps other regions. Regions have to be unique.");
        }
    }

    // Validate that all of the regions are within the range of the input sequences.
    std::unordered_map<std::string, int64_t> len_dict;
    for (const auto& [key, val] : seq_lens) {
        len_dict[key] = val;
    }
    for (const auto& region : regions) {
        const auto it = len_dict.find(region.name);
        if (it == std::end(len_dict)) {
            throw std::runtime_error{"Region validation failed: sequence name for region '" +
                                     polisher::region_to_string(region) +
                                     "' does not exist in the input sequence file."};
        }
        const int64_t seq_len = it->second;
        // Allow negative coordinates as a proxy for full sequence length.
        if ((region.start >= seq_len) || (region.end > seq_len) ||
            ((region.start >= 0) && (region.end >= 0) && (region.start >= region.end))) {
            throw std::runtime_error{
                    "Region validation failed: coordinates for region '" +
                    polisher::region_to_string(region) +
                    "' are not valid. Sequence length: " + std::to_string(seq_len)};
        }
    }
}

std::pair<std::vector<std::vector<polisher::Region>>, std::vector<polisher::Interval>>
prepare_region_batches(const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                       const std::vector<polisher::Region>& user_regions,
                       const int64_t draft_batch_size) {
    // Create a lookup.
    std::unordered_map<std::string, int64_t> draft_ids;
    for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
        draft_ids[draft_lens[seq_id].first] = seq_id;
    }

    // Outer vector: ID of the draft, inner vector: regions.
    std::vector<std::vector<polisher::Region>> ret(std::size(draft_lens));

    if (std::empty(user_regions)) {
        // Add full draft sequences.
        for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
            const auto& [draft_name, draft_len] = draft_lens[seq_id];
            ret[seq_id].emplace_back(polisher::Region{draft_name, 0, draft_len});
        }

    } else {
        // Bin the user regions for individual contigs.
        for (const auto& region : user_regions) {
            const auto it = draft_ids.find(region.name);
            if (it == std::end(draft_ids)) {
                throw std::runtime_error(
                        "Sequence name from a custom specified region not found in the input "
                        "sequence file! region: " +
                        polisher::region_to_string(region));
            }
            const int64_t seq_id = it->second;
            ret[seq_id].emplace_back(polisher::Region{region.name, region.start, region.end});
        }
    }

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    std::vector<polisher::Interval> region_batches = polisher::create_batches(
            ret, draft_batch_size, [](const std::vector<polisher::Region>& regions) {
                int64_t sum = 0;
                for (const auto& region : regions) {
                    sum += region.end - region.start;
                }
                return sum;
            });

    return std::make_pair(std::move(ret), std::move(region_batches));
}

}  // namespace

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
            load_seq_lengths(opt.in_draft_fastx_fn);

    // Create windows only for the selected regions.
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> draft_lookup;
    for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
        draft_lookup[draft_lens[seq_id].first] = {seq_id, draft_lens[seq_id].second};
    }

    validate_regions(opt.regions, draft_lens);

    // Open the draft FASTA file.
    const hts_io::FastxRandomReader draft_reader(opt.in_draft_fastx_fn);

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
    auto ofs_consensus = get_output_stream(out_consensus_fn);

    // Open the output stream to a file/stdout for the variant calls.
    const std::filesystem::path out_vcf_fn =
            (std::empty(opt.output_dir)) ? "" : (opt.output_dir / "variants.vcf");
    auto ofs_vcf = get_output_stream(out_vcf_fn);

    // Prepare regions for processing.
    const auto [input_regions, region_batches] =
            prepare_region_batches(draft_lens, opt.regions, opt.draft_batch_size);

    // Update the progress tracker.
    {
        const int64_t total_input_bases =
                std::accumulate(std::begin(input_regions), std::end(input_regions),
                                static_cast<int64_t>(0), [](const int64_t a, const auto& b) {
                                    int64_t sum = 0;
                                    for (const auto& region : b) {
                                        sum += region.end - region.start;
                                    }
                                    return a + sum;
                                });
        polish_stats.update("total", static_cast<double>(total_input_bases));
        polish_stats.update("processed", 0.0);
    }

    // Process the draft sequences in batches of user-specified size.
    for (const auto& batch_interval : region_batches) {
        // Get the regions for this interval.
        std::vector<polisher::Region> region_batch;
        for (int32_t i = batch_interval.start; i < batch_interval.end; ++i) {
            region_batch.insert(std::end(region_batch), std::begin(input_regions[i]),
                                std::end(input_regions[i]));
        }

        const int64_t batch_bases = std::accumulate(
                std::begin(region_batch), std::end(region_batch), static_cast<int64_t>(0),
                [](const int64_t a, const auto& b) { return a + b.end - b.start; });

        // Debug print.
        spdlog::debug("[run_polishing] =============================");
        spdlog::debug("[run_polishing] Processing batch interval of drafts: [{}, {})",
                      batch_interval.start, batch_interval.end);
        for (int64_t i = 0; i < dorado::ssize(region_batch); ++i) {
            spdlog::debug("[run_polishing] region_batch i = {}: {}", i,
                          polisher::region_to_string(region_batch[i]));
        }

        // Split the sequences into larger BAM windows, like Medaka.
        // NOTE: the window.seq_id is the _absolute_ sequence ID of the input draft sequences.
        spdlog::debug("Creating BAM windows.");
        const std::vector<polisher::Window> bam_regions = polisher::create_windows_from_regions(
                region_batch, draft_lookup, opt.bam_chunk, opt.window_overlap);

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
        std::vector<polisher::ConsensusResult> all_results_cons;
        std::vector<polisher::VariantCallingSample> vc_input_data;

        std::thread thread_sample_producer =
                std::thread(&polisher::sample_producer, std::ref(resources), std::cref(bam_regions),
                            std::cref(draft_lens), opt.threads, opt.batch_size, opt.window_len,
                            opt.window_overlap, opt.bam_subchunk, std::ref(batch_queue));

        std::thread thread_sample_decoder =
                std::thread(&polisher::decode_samples_in_parallel, std::ref(all_results_cons),
                            std::ref(vc_input_data), std::ref(decode_queue), std::ref(polish_stats),
                            std::cref(*resources.decoder), opt.threads, opt.min_depth);

        polisher::infer_samples_in_parallel(batch_queue, decode_queue, resources.models,
                                            *resources.encoder);

        if (thread_sample_producer.joinable()) {
            thread_sample_producer.join();
        }

        if (thread_sample_decoder.joinable()) {
            thread_sample_decoder.join();
        }

        spdlog::debug(
                "[run_polishing] Stitching sequences: {}-{}/{} (number: {}, total "
                "length: {:.2f} Mbp), parts: {}",
                batch_interval.start, batch_interval.end, std::size(input_regions),
                std::size(region_batch), batch_bases / (1000.0 * 1000.0),
                std::size(all_results_cons));

        spdlog::debug(
                "Data for variant calling: vc_input_data.size() = {}, all_results_cons.size() = {}",
                std::size(vc_input_data), std::size(all_results_cons));

        // Construct the consensus sequences, only if they will be written.
        if (opt.write_consensus) {
            const std::vector<std::vector<polisher::ConsensusResult>> consensus_seqs =
                    construct_consensus_seqs(batch_interval, all_results_cons, draft_reader,
                                             draft_lens, opt.fill_gaps, opt.fill_char);

            // Write the consensus file.
            for (const auto& consensus : consensus_seqs) {
                write_consensus_results(*ofs_consensus, consensus, opt.fill_gaps,
                                        (opt.out_format == OutputFormat::FASTQ));
            }
        }

        // Run variant calling, optionally.
        if (opt.run_variant_calling) {
            const std::vector<polisher::Variant> variants = call_variants(
                    batch_interval, vc_input_data, draft_reader, draft_lens, *resources.decoder,
                    opt.ambig_ref, opt.vc_type == VariantCallingEnum::GVCF);

            // Write the VCF file.
            // clang-format off
            *ofs_vcf << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
            for (const auto& var : variants) {
                *ofs_vcf << draft_lens[var.seq_id].first
                     << '\t' << (var.pos + 1)
                     << '\t' << '.'
                     << '\t' << var.ref
                     << '\t' << var.alt
                     << '\t' << var.qual
                     << '\t' << var.filter
                     << '\t' << '.'
                     << '\n';
            }
            // clang-format on
        }
    }
}

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
        const polisher::BamInfo bam_info = polisher::analyze_bam(opt.in_aln_bam_fn, opt.read_group);

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
            check_read_groups(bam_info, opt.read_group);
        }

        // Set the number of threads so that libtorch doesn't cause a thread bomb.
        // at::set_num_interop_threads(opt.threads);
        torch::set_num_threads(1);

        // Resolve the model for polishing.
        const polisher::ModelConfig model_config = resolve_model(
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
