#include "cli/cli_utils.h"
#include "correct/CorrectionProgressTracker.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "read_pipeline/CorrectionNode.h"
#include "read_pipeline/ErrorCorrectionMapperNode.h"
#include "read_pipeline/HtsWriter.h"
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

using OutputMode = dorado::utils::HtsFile::OutputMode;

int correct(int argc, char* argv[]) {
    utils::make_torch_deterministic();
    utils::arg_parse::ArgParser parser("dorado correct");
    parser.visible.add_description("Dorado read correction tool.");
    parser.visible.add_argument("reads").help(
            "Path to a file with reads to correct in FASTQ format.");
    parser.visible.add_argument("-t", "--threads")
            .help("Number of threads for processing. "
                  "Default uses "
                  "all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("--infer-threads")
            .help("Number of threads per device.")
#if DORADO_CUDA_BUILD
            .default_value(2)
#else
            .default_value(1)
#endif
            .scan<'i', int>();
    parser.visible.add_argument("-b", "--batch-size")
            .help("Batch size for inference. Default: 0 for auto batch size detection.")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("-i", "--index-size")
            .help("Size of index for mapping and alignment. Default 8G. Decrease index size to "
                  "lower memory footprint.")
            .default_value(std::string{"8G"});
    parser.visible.add_argument("-m", "--model-path").help("Path to correction model folder.");
    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();
    cli::add_device_arg(parser);

    try {
        utils::arg_parse::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.visible.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.visible.get<std::vector<std::string>>("reads"));
    auto threads(parser.visible.get<int>("threads"));
    auto infer_threads(parser.visible.get<int>("infer-threads"));
    auto device(parser.visible.get<std::string>("device"));
    if (!cli::validate_device_string(device)) {
        return EXIT_FAILURE;
    }
    if (device == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        device = "cpu";
#else
        device = cli::get_auto_detected_device();
#endif
    }

    auto batch_size(parser.visible.get<int>("batch-size"));
    auto index_size(utils::arg_parse::parse_string_to_size<uint64_t>(
            parser.visible.get<std::string>("index-size")));

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    const int aligner_threads = threads;
    const int correct_threads = std::max(4, static_cast<int>(threads / 4));
    const int correct_writer_threads = 1;
    spdlog::debug("> aligner threads {}, corrector threads {}, writer threads {}", aligner_threads,
                  correct_threads, correct_writer_threads);

    if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        std::exit(EXIT_FAILURE);
    }

    if (!std::filesystem::exists(reads.front())) {
        spdlog::error("Input reads file {} does not exist!", reads.front());
        std::exit(EXIT_FAILURE);
    }

    std::filesystem::path model_dir;
    bool remove_tmp_dir = false;
    if (parser.visible.is_used("--model-path")) {
        model_dir = std::filesystem::path(parser.visible.get<std::string>("model-path"));

        if (!std::filesystem::exists(model_dir)) {
            spdlog::error("Input model path {} does not exist!", model_dir.string());
            std::exit(EXIT_FAILURE);
        }
    } else {
        // Download model
        auto tmp_dir = utils::get_downloads_path(std::nullopt);
        const std::string model_name = "herro-v1";
        auto success = model_downloader::download_models(tmp_dir.string(), model_name);
        if (!success) {
            spdlog::error("Could not download model: {}", model_name);
            std::exit(EXIT_FAILURE);
        }
        model_dir = (tmp_dir / "herro-v1");
        remove_tmp_dir = true;
    }

    // The overall pipeline will be as follows -
    // 1. The Alignment node will be responsible
    // for running all-vs-all alignment. Since the index
    // for a full genome could be larger than what can
    // fit in memory, the alignment node needs to support
    // split mm2 indices. This requires the input reads
    // to be iterated over multiple times, once for each
    // index chunk. In order to manage this properly, the
    // input file reading is handled within the alignment node.
    // Each alignment out of that node is expected to generate
    // multiple aligned records, which will all be packaged
    // and sent into a window generation node.
    // 2. Correction node will chunk up the alignments into
    // multiple windows, create tensors, run inference and decode
    // the windows into a final corrected sequence.
    // 3. Corrected reads will be written out FASTA or BAM format.

    // Setup outut file.
    auto output_mode = OutputMode::FASTA;
    utils::HtsFile hts_file("-", output_mode, correct_writer_threads, false);

    PipelineDescriptor pipeline_desc;
    // 3. Corrected reads will be written out FASTA or BAM format.
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, "");

    // 2. Window generation, encoding + inference and decoding to generate
    // final reads.
    pipeline_desc.add_node<CorrectionNode>({hts_writer}, reads[0], correct_threads, device,
                                           infer_threads, batch_size, model_dir);

    // 1. Alignment node that generates alignments per read to be
    // corrected.
    ErrorCorrectionMapperNode aligner(reads[0], aligner_threads, index_size);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        return EXIT_FAILURE;
    }

    // Set up stats counting.
    CorrectionProgressTracker tracker;
    tracker.set_description("Correcting");
    std::vector<dorado::stats::StatsCallable> stats_callables;
    // Aligner stats need to be passed separately since the aligner node
    // is not part of the pipeline, so the stats are not automatically
    // gathered.
    stats_callables.push_back([&tracker, &aligner](const stats::NamedStats& stats) {
        tracker.update_progress_bar(stats, aligner.sample_stats());
    });
    constexpr auto kStatsPeriod = 1000ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
    // End stats counting setup.

    spdlog::info("> starting correction");
    // Start the pipeline.
    aligner.process(*pipeline);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats, aligner.sample_stats());

    // Report progress during output file finalisation.
    hts_file.finalise([&](size_t) {});
    tracker.summarize();

    spdlog::info("> finished correction");

    if (remove_tmp_dir) {
        std::filesystem::remove_all(model_dir.parent_path());
    }

    return 0;
}

}  // namespace dorado
