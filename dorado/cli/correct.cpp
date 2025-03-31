#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "correct/CorrectionProgressTracker.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "read_pipeline/CorrectionInferenceNode.h"
#include "read_pipeline/CorrectionMapperNode.h"
#include "read_pipeline/CorrectionPafReaderNode.h"
#include "read_pipeline/CorrectionPafWriterNode.h"
#include "read_pipeline/HtsWriter.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/torch_utils.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/string_utils.h"

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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

namespace dorado {

namespace {

using OutputMode = dorado::utils::HtsFile::OutputMode;
using ParserPtr = std::unique_ptr<utils::arg_parse::ArgParser>;

/// \brief All options for the Dorado Correct tool.
struct Options {
    std::vector<std::string> in_reads_fns;
    int verbosity = 0;
    int threads = 0;
    int infer_threads = 0;
    std::string device;
    int batch_size = 0;
    uint64_t index_size = 0;
    bool to_paf = false;
    std::string in_paf_fn;
    std::string model_path;
    std::string resume_path_fn;
    bool compute_num_blocks = false;
    std::optional<int> run_block_id;
    int32_t kmer_size = 25;
    int32_t ovl_window_size = 17;
    int32_t min_chain_score = 2500;
    float mid_occ_frac = 0.05f;
    std::unordered_set<std::string> debug_tnames;
    bool legacy_windowing = false;
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado correct");

    parser->visible.add_description("Dorado read correction tool");

    {
        // Positional arguments group
        parser->visible.add_argument("reads").help(
                "Path to a file with reads to correct in FASTQ format.");
    }
    {
        // Default "Optional arguments" group
        parser->visible.add_argument("-t", "--threads")
                .help("Number of threads for processing. "
                      "Default uses all available threads.")
                .default_value(0)
                .scan<'i', int>();
        parser->visible.add_argument("--infer-threads")
                .help("Number of threads per device.")
#if DORADO_CUDA_BUILD
                .default_value(3)
#else
                .default_value(1)
#endif
                .scan<'i', int>();

        cli::add_device_arg(*parser);

        // Default "Optional arguments" group
        parser->visible.add_argument("-v", "--verbose")
                .default_value(false)
                .implicit_value(true)
                .nargs(0)
                .action([&](const auto&) { ++verbosity; })
                .append();
    }
    {
        parser->visible.add_group("Input/output arguments");
        parser->visible.add_argument("-m", "--model-path").help("Path to correction model folder.");
        parser->visible.add_argument("-p", "--from-paf")
                .help("Path to a PAF file with alignments. Skips alignment computation.");
        parser->visible.add_argument("--to-paf")
                .help("Generate PAF alignments and skip consensus.")
                .default_value(false)
                .implicit_value(true);
        parser->visible.add_argument("--resume-from")
                .help("Resume a previously interrupted run. Requires a path to a file where "
                      "sequence headers are stored in the first column (whitespace delimited), one "
                      "per row. The header can also occupy a full row with no other columns. For "
                      "example, a .fai index generated from the previously corrected output FASTA "
                      "file is a valid input here.")
                .default_value("");
    }
    {
        parser->visible.add_group("Advanced arguments");
        parser->visible.add_argument("-b", "--batch-size")
                .help("Batch size for inference. Default: 0 for auto batch size detection.")
                .default_value(0)
                .scan<'i', int>();
        parser->visible.add_argument("-i", "--index-size")
                .help("Size of index for mapping and alignment. Default 8G. Decrease index size to "
                      "lower memory footprint.")
                .default_value(std::string{"8G"});
        parser->visible.add_argument("--compute-num-blocks")
                .help("Computes and returns one number: the number of index blocks which would be "
                      "processed on a normal run.")
                .flag();
        parser->visible.add_argument("--run-block-id")
                .help("ID of the index block to run. If specified, only this block will be run.")
                .scan<'i', int>();
    }
    {
        parser->hidden.add_argument("--kmer-size")
                .help("Minimizer kmer size for overlapping.")
                .default_value(25)
                .scan<'i', int>();
        parser->hidden.add_argument("--ovl-window-size")
                .help("Minimizer window size score for overlapping.")
                .default_value(17)
                .scan<'i', int>();
        parser->hidden.add_argument("--min-chain-score")
                .help("Minimum chaining score for overlapping.")
                .default_value(2500)
                .scan<'i', int>();
        parser->hidden.add_argument("--mid-occ-frac")
                .help("Filter out top FLOAT fraction of repetitive minimizers during the overlap "
                      "process.")
                .default_value(0.05f)
                .scan<'g', float>();
        parser->hidden.add_argument("--debug-tnames")
                .help("Comma separated list of one or more target read names to process.")
                .default_value("");
        parser->hidden.add_argument("--legacy-windowing")
                .help("Runs legacy windowing instead of the new version.")
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

    opt.in_reads_fns = parser.visible.get<std::vector<std::string>>("reads");
    opt.infer_threads = parser.visible.get<int>("infer-threads");
    opt.batch_size = parser.visible.get<int>("batch-size");
    opt.index_size = std::max<int64_t>(0, utils::arg_parse::parse_string_to_size<int64_t>(
                                                  parser.visible.get<std::string>("index-size")));
    opt.to_paf = parser.visible.get<bool>("to-paf");
    opt.in_paf_fn = (parser.visible.is_used("--from-paf"))
                            ? parser.visible.get<std::string>("from-paf")
                            : "";
    opt.resume_path_fn = parser.visible.get<std::string>("resume-from");
    opt.model_path = (parser.visible.is_used("--model-path"))
                             ? parser.visible.get<std::string>("model-path")
                             : "";

    opt.threads = parser.visible.get<int>("threads");
    opt.threads = (opt.threads == 0) ? std::thread::hardware_concurrency() : opt.threads;

    opt.device = parser.visible.get<std::string>("device");

    if (opt.device == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        opt.device = "cpu";
#else
        opt.device = utils::get_auto_detected_device();
#endif
    }

    opt.verbosity = verbosity;

    opt.compute_num_blocks = parser.visible.get<bool>("compute-num-blocks");
    opt.run_block_id = parser.visible.present<int>("run-block-id");

    // Hidden parameters.
    opt.kmer_size = parser.hidden.get<int>("kmer-size");
    opt.ovl_window_size = parser.hidden.get<int>("ovl-window-size");
    opt.min_chain_score = parser.hidden.get<int>("min-chain-score");
    opt.mid_occ_frac = parser.hidden.get<float>("mid-occ-frac");
    opt.legacy_windowing = parser.hidden.get<bool>("legacy-windowing");

    // Debug option, hidden.
    const std::string tnames_str = parser.hidden.get<std::string>("debug-tnames");
    if (!std::empty(tnames_str)) {
        const std::vector<std::string> tnames = utils::split(tnames_str, ',');
        opt.debug_tnames.insert(std::begin(tnames), std::end(tnames));
    }

    return opt;
}

std::filesystem::path download_model(const std::string& model_name) {
    const std::filesystem::path tmp_dir = utils::get_downloads_path(std::nullopt);
    const bool success = model_downloader::download_models(tmp_dir.string(), model_name);
    if (!success) {
        spdlog::error("Could not download model: {}", model_name);
        std::exit(EXIT_FAILURE);
    }
    return (tmp_dir / "herro-v1");
}

struct HtsFileDeleter {
    void operator()(utils::HtsFile* ptr) const {
        if (ptr) {
            ptr->finalise([&](size_t) {});
        }
        delete ptr;
    }
};

std::unordered_set<std::string> load_processed_reads(const std::filesystem::path& in_path) {
    std::unordered_set<std::string> ret;
    std::string line;
    std::string token;
    std::ifstream ifs(in_path);
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            continue;
        }

        // Split on whitespace and ':'. The ':' is important because Dorado Correct
        // can generate multiple output reads from the same input read, and it adds
        // a ":<num>" suffix to the header in these cases.
        const std::size_t found = line.find_first_of(": \t");
        std::string header = line.substr(0, found);

        // Check for malformed inputs.
        if (std::empty(header)) {
            throw std::runtime_error{"Found empty string in the first column of input file: " +
                                     in_path.string() + ", line: '" + line + "'."};
        }

        ret.emplace(std::move(header));
    }
    return ret;
}

std::tuple<std::string, int64_t> find_furthest_skipped_read(
        const std::filesystem::path& in_fastx_fn,
        const std::unordered_set<std::string>& skip_set) {
    if (std::empty(skip_set)) {
        return std::tuple(std::string(), -1);
    }

    const bool rv_fai = utils::create_fai_index(in_fastx_fn);
    if (!rv_fai) {
        spdlog::error("Failed to create/verify a .fai index for input file: '{}'!",
                      in_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    const std::filesystem::path in_reads_fai_fn = utils::get_fai_path(in_fastx_fn);

    std::string furthest_reach;
    int64_t furthest_reach_id{-1};
    int64_t num_loaded{0};

    std::ifstream ifs(in_reads_fai_fn);
    std::string line;
    std::string token;
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            throw std::runtime_error{"Empty lines found in the input index file: " +
                                     in_reads_fai_fn.string()};
        }

        // Split on whitespace and ':'. The ':' is important because Dorado Correct
        // can generate multiple output reads from the same input read, and it adds
        // a ":<num>" suffix to the header in these cases.
        const std::size_t found = line.find_first_of(": \t");
        std::string header = line.substr(0, found);

        // Check for malformed inputs.
        if (std::empty(header)) {
            throw std::runtime_error{
                    "Found empty string in the first column of input index file: " +
                    in_reads_fai_fn.string() + ", line: '" + line + "'."};
        }

        if (skip_set.count(header) > 0) {
            std::swap(furthest_reach, header);
            furthest_reach_id = num_loaded;
        }
        ++num_loaded;
    }

    return std::tuple(furthest_reach, furthest_reach_id);
}

void validate_options(const Options& opt) {
    // Parameter validation.
    if (!cli::validate_device_string(opt.device)) {
        std::exit(EXIT_FAILURE);
    }
    if (opt.in_reads_fns.size() > 1) {
        spdlog::error("Multi file input not yet handled");
        std::exit(EXIT_FAILURE);
    }
    if (std::empty(opt.in_reads_fns)) {
        spdlog::error("At least one input reads file needs to be specified.");
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_reads_fns.front())) {
        spdlog::error("Input reads file {} does not exist!", opt.in_reads_fns.front());
        std::exit(EXIT_FAILURE);
    }
    if (!std::empty(opt.in_paf_fn) && !std::filesystem::exists(opt.in_paf_fn)) {
        spdlog::error("Input PAF path {} does not exist!", opt.in_paf_fn);
        std::exit(EXIT_FAILURE);
    }
    if (!std::empty(opt.model_path) && !std::filesystem::exists(opt.model_path)) {
        spdlog::error("Input model directory {} does not exist!", opt.model_path);
        std::exit(EXIT_FAILURE);
    }
    if (!std::empty(opt.resume_path_fn) && !std::filesystem::exists(opt.resume_path_fn)) {
        spdlog::error("Input resume index file {} does not exist!", opt.resume_path_fn);
        std::exit(EXIT_FAILURE);
    }
    if (opt.compute_num_blocks && !std::empty(opt.resume_path_fn)) {
        spdlog::error("The --compute-num-blocks option cannot be used together with --resume-from.",
                      opt.resume_path_fn);
        std::exit(EXIT_FAILURE);
    }
    if ((opt.run_block_id) && (*opt.run_block_id < 0)) {
        spdlog::error("The --run-block-id option cannot be negative.");
        std::exit(EXIT_FAILURE);
    }
    if ((opt.run_block_id) && (*opt.run_block_id >= 0) && !std::empty(opt.resume_path_fn)) {
        spdlog::error("The --run-block-id option cannot be used together with --resume-from.",
                      opt.resume_path_fn);
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int correct(int argc, char* argv[]) {
    utils::make_torch_deterministic();

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

    // Initialize the log level.
    if (opt.verbosity) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(opt.verbosity));
    }

    // Check if input options are good.
    validate_options(opt);

    // Set the number of threads so that libtorch doesn't cause a thread bomb.
    utils::initialise_torch();

    // After validation, there is exactly one reads file allowed:
    const std::string in_reads_fn = opt.in_reads_fns.front();

    // Compute the number of threads for each stage.
    const int aligner_threads = opt.threads;
    const int correct_threads =
            (!std::empty(opt.debug_tnames)) ? 1 : std::max(4, static_cast<int>(opt.threads / 4));
    const int correct_writer_threads = 1;
    spdlog::debug("Aligner threads {}, corrector threads {}, writer threads {}", aligner_threads,
                  correct_threads, correct_writer_threads);

    // If model dir is not specified, download the model.
    const auto [model_dir, remove_tmp_dir] = [&opt]() {
        std::filesystem::path ret_model_dir = opt.model_path;
        bool ret_remove_tmp_dir = false;
        if (!opt.to_paf && std::empty(ret_model_dir)) {
            ret_model_dir = download_model("herro-v1");
            ret_remove_tmp_dir = true;
        }
        return std::tuple(ret_model_dir, ret_remove_tmp_dir);
    }();

    // Load the resume list if it exists.
    try {
        if (opt.compute_num_blocks) {
            spdlog::debug("Only computing the number of index blocks.");

            CorrectionMapperNode node(in_reads_fn, aligner_threads, opt.index_size, {}, {}, -1,
                                      opt.kmer_size, opt.ovl_window_size, opt.min_chain_score,
                                      opt.mid_occ_frac);

            // Loop through all index blocks.
            while (node.load_next_index_block()) {
            }

            // Handle empty sequence files.
            const int64_t num_blocks =
                    node.get_current_index_block_id() + ((node.get_index_seqs() > 0) ? 1 : 0);

            std::cout << num_blocks << std::endl;

            return EXIT_SUCCESS;
        }

        std::unordered_set<std::string> skip_set =
                (!std::empty(opt.resume_path_fn)) ? load_processed_reads(opt.resume_path_fn)
                                                  : std::unordered_set<std::string>{};

        // Find the furthest processed input read in the skip set. If skip_set is empty,
        const auto [furthest_skip_header, furthest_skip_id] =
                find_furthest_skipped_read(in_reads_fn, skip_set);
        spdlog::debug("furthest_skip_header = '{}', furthest_skip_id = {}", furthest_skip_header,
                      furthest_skip_id);

        // Workflow construction.
        // The overall pipeline will be as follows:
        //  1. The Alignment node will be responsible for running all-vs-all alignment.
        //      Since the MM2 index for a full genome could be larger than what can
        //      fit in memory, the alignment node needs to support split mm2 indices.
        //      This requires the input reads to be iterated over multiple times, once
        //      for each index chunk. In order to manage this properly, the input file
        //      reading is handled within the alignment node.
        //      Each alignment out of that node is expected to generate multiple aligned
        //      records, which will all be packaged and sent into a window generation node.
        //  2. Correction node will chunk up the alignments into multiple windows, create
        //      tensors, run inference and decode the windows into a final corrected sequence.
        //  3. Corrected reads will be written out FASTA or BAM format.

        std::unique_ptr<utils::HtsFile, HtsFileDeleter> hts_file;
        PipelineDescriptor pipeline_desc;

        // Add the writer node and (optionally) the correction node.
        if (!opt.to_paf) {
            // Setup output file.
            hts_file = std::unique_ptr<utils::HtsFile, HtsFileDeleter>(
                    new utils::HtsFile("-", OutputMode::FASTA, correct_writer_threads, false));

            // 3. Corrected reads will be written out in FASTA format.
            const NodeHandle hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, "");

            // 2. Window generation, encoding + inference and decoding to generate final reads.
            pipeline_desc.add_node<CorrectionInferenceNode>(
                    {hts_writer}, in_reads_fn, correct_threads, opt.device, opt.infer_threads,
                    opt.batch_size, model_dir, opt.legacy_windowing, opt.debug_tnames);
        } else {
            pipeline_desc.add_node<CorrectionPafWriterNode>({});
        }

        // Create the Pipeline from our description.
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
        if (!pipeline) {
            spdlog::error("Failed to create pipeline");
            return EXIT_FAILURE;
        }

        // Create the entry (input) node (either the mapper or the reader).
        // Aligner stats need to be passed separately since the aligner node
        // is not part of the pipeline, so the stats are not automatically gathered.
        std::unique_ptr<MessageSink> aligner;
        if (!std::empty(opt.in_paf_fn)) {
            aligner = std::make_unique<CorrectionPafReaderNode>(opt.in_paf_fn, std::move(skip_set));
        } else {
            // 1. Alignment node that generates alignments per read to be corrected.
            aligner = std::make_unique<CorrectionMapperNode>(
                    in_reads_fn, aligner_threads, opt.index_size, furthest_skip_header,
                    std::move(skip_set), (opt.run_block_id) ? *opt.run_block_id : -1, opt.kmer_size,
                    opt.ovl_window_size, opt.min_chain_score, opt.mid_occ_frac);
        }

        // Set up stats counting.
        CorrectionProgressTracker tracker;
        tracker.set_description("Correcting");
        std::vector<dorado::stats::StatsCallable> stats_callables;
        stats_callables.push_back([&tracker, &aligner](const stats::NamedStats& stats) {
            tracker.update_progress_bar(stats, aligner->sample_stats());
        });
        constexpr auto kStatsPeriod = 1000ms;
        auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
        // End stats counting setup.

        spdlog::info("Starting");

        // Start the pipeline.
        if (!std::empty(opt.in_paf_fn)) {
            dynamic_cast<CorrectionPafReaderNode*>(aligner.get())->process(*pipeline);
        } else {
            dynamic_cast<CorrectionMapperNode*>(aligner.get())->process(*pipeline);
        }

        // Wait for the pipeline to complete.  When it does, we collect
        // final stats to allow accurate summarisation.
        auto final_stats = pipeline->terminate(DefaultFlushOptions());
        stats_sampler->terminate();
        tracker.update_progress_bar(final_stats, aligner->sample_stats());

        // Report progress during output file finalisation.
        tracker.summarize();

        spdlog::info("Finished");

        if (remove_tmp_dir) {
            std::filesystem::remove_all(model_dir.parent_path());
        }

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}.", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        spdlog::error("Caught an unknown exception!");
        return EXIT_FAILURE;
    }

    return 0;
}

}  // namespace dorado
