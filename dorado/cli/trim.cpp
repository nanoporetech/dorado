#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "demux/adapter_info.h"
#include "dorado_version.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/TrimmerNode.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
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

int trim(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description("Adapter/primer trimming tool.");
    parser.add_argument("reads")
            .help("Path to a file with reads to trim. Can be in any HTS format.")
            .nargs(argparse::nargs_pattern::any);
    parser.add_argument("-t", "--threads")
            .help("Combined number of threads for adapter/primer detection and output generation. "
                  "Default uses all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("Maximum number of reads to process. Mainly for debugging. Process all reads by "
                  "default.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-k", "--sequencing-kit")
            .help("Sequencing kit name to use for selecting adapters and primers to trim.");
    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to trim.")
            .default_value(std::string(""));
    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();
    parser.add_argument("--emit-fastq")
            .help("Output in fastq format. Default is BAM.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--no-trim-primers")
            .help("Skip primer detection and trimming. Only adapters will be detected and trimmed.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--primer-sequences").help("Path to file with custom primer sequences.");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    if (!parser.is_used("--sequencing-kit")) {
        spdlog::error("The sequencing kit name must be specified with --sequencing-kit.");
        return EXIT_FAILURE;
    }
    auto kit_name = parser.get<std::string>("--sequencing-kit");
    if (kit_name.empty()) {
        spdlog::error("Sequencing kit name must be non-empty.");
        return EXIT_FAILURE;
    }

    auto reads(parser.get<std::vector<std::string>>("reads"));
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));
    std::vector<std::string> args(argv, argv + argc);

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // adapter/primer trimming. Heuristically use 10% of threads for BAM
    // generation and rest for trimming.
    auto [trim_threads, trim_writer_threads] =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> adapter/primer trimming threads {}, writer threads {}", trim_threads,
                  trim_writer_threads);

    auto read_list = utils::load_read_list(parser.get<std::string>("--read-ids"));

    if (reads.empty()) {
        if (utils::is_fd_tty(stdin)) {
            std::cout << parser << '\n';
            return EXIT_FAILURE;
        }
        reads.push_back("-");
    } else if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        return EXIT_FAILURE;
    }

    HtsReader reader(reads[0], read_list);
    auto header = SamHdrPtr(sam_hdr_dup(reader.header()));
    cli::add_pg_hdr(header.get(), "trim", args, "cpu");
    // Always remove alignment information from input header
    // because at minimum the adapters are trimmed, which
    // invalidates the alignment record.
    utils::strip_alignment_data_from_header(header.get());

    auto output_mode = OutputMode::BAM;

    auto emit_fastq = parser.get<bool>("--emit-fastq");

    if (emit_fastq) {
        spdlog::info(" - Note: FASTQ output is not recommended as not all data can be preserved.");
        output_mode = OutputMode::FASTQ;
    } else if (utils::is_fd_tty(stdout)) {
        output_mode = OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = OutputMode::UBAM;
    }

    std::optional<std::string> custom_primer_file = std::nullopt;
    if (parser.is_used("--primer-sequences")) {
        custom_primer_file = parser.get<std::string>("--primer-sequences");
    }

    utils::HtsFile hts_file("-", output_mode, trim_writer_threads, false);
    hts_file.set_header(header.get());

    PipelineDescriptor pipeline_desc;
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, "");

    auto trimmer = pipeline_desc.add_node<TrimmerNode>({hts_writer}, 1);

    auto adapter_info = std::make_shared<demux::AdapterInfo>();
    adapter_info->trim_adapters = true;
    adapter_info->trim_primers = !parser.get<bool>("--no-trim-primers");
    adapter_info->kit_name = kit_name;
    adapter_info->custom_seqs = custom_primer_file;

    auto client_info = std::make_shared<DefaultClientInfo>();
    client_info->contexts().register_context<const demux::AdapterInfo>(adapter_info);
    reader.set_client_info(client_info);

    pipeline_desc.add_node<AdapterDetectorNode>({trimmer}, trim_threads);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        return EXIT_FAILURE;
    }

    // Set up stats counting
    ProgressTracker tracker(ProgressTracker::Mode::TRIM, 0,
                            hts_file.finalise_is_noop() ? 0.f : 0.5f);
    tracker.set_description("Trimming");
    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
    // End stats counting setup.

    spdlog::info("> starting adapter/primer trimming");
    reader.read(*pipeline, max_reads);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);

    // Report progress during output file finalisation.
    tracker.set_description("Sorting output files");
    hts_file.finalise([&](size_t progress) {
        tracker.update_post_processing_progress(static_cast<float>(progress));
    });
    tracker.summarize();

    spdlog::info("> finished adapter/primer trimming");

    return EXIT_SUCCESS;
}

}  // namespace dorado
