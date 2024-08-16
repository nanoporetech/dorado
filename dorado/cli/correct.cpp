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
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
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
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado correct");
    parser->visible.add_description("Dorado read correction tool.");
    parser->visible.add_argument("reads").help(
            "Path to a file with reads to correct in FASTQ format.");
    parser->visible.add_argument("-t", "--threads")
            .help("Number of threads for processing. "
                  "Default uses "
                  "all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser->visible.add_argument("--infer-threads")
            .help("Number of threads per device.")
#if DORADO_CUDA_BUILD
            .default_value(2)
#else
            .default_value(1)
#endif
            .scan<'i', int>();
    parser->visible.add_argument("-b", "--batch-size")
            .help("Batch size for inference. Default: 0 for auto batch size detection.")
            .default_value(0)
            .scan<'i', int>();
    parser->visible.add_argument("-i", "--index-size")
            .help("Size of index for mapping and alignment. Default 8G. Decrease index size to "
                  "lower memory footprint.")
            .default_value(std::string{"8G"});
    parser->visible.add_argument("-m", "--model-path").help("Path to correction model folder.");
    parser->visible.add_argument("-p", "--from-paf")
            .help("Path to a PAF file with alignments. Skips alignment computation.");
    parser->visible.add_argument("--to-paf")
            .help("Generate PAF alignments and skip consensus.")
            .default_value(false)
            .implicit_value(true);
    parser->visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();
    cli::add_device_arg(*parser);

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
    opt.index_size = utils::arg_parse::parse_string_to_size<uint64_t>(
            parser.visible.get<std::string>("index-size"));
    opt.to_paf = parser.visible.get<bool>("to-paf");
    opt.in_paf_fn = (parser.visible.is_used("--from-paf"))
                            ? parser.visible.get<std::string>("from-paf")
                            : "";
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

    // Parameter validation.
    if (!cli::validate_device_string(opt.device)) {
        return EXIT_FAILURE;
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

    // Compute the number of threads for each stage.
    const int aligner_threads = opt.threads;
    const int correct_threads = std::max(4, static_cast<int>(opt.threads / 4));
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
                {hts_writer}, opt.in_reads_fns[0], correct_threads, opt.device, opt.infer_threads,
                opt.batch_size, model_dir);
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
        aligner = std::make_unique<CorrectionPafReaderNode>(opt.in_paf_fn);
    } else {
        // 1. Alignment node that generates alignments per read to be corrected.
        aligner = std::make_unique<CorrectionMapperNode>(opt.in_reads_fns.front(), aligner_threads,
                                                         opt.index_size);
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

    return 0;
}

}  // namespace dorado
