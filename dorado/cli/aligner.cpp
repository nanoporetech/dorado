#include "Version.h"
#include "alignment/IndexFileAccess.h"
#include "cli/cli_utils.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"

#include <minimap.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

namespace dorado {

void add_pg_hdr(sam_hdr_t* hdr) {
    sam_hdr_add_line(hdr, "PG", "ID", "aligner", "PN", "dorado", "VN", DORADO_VERSION, "DS",
                     MM_VERSION, NULL);
}

int aligner(int argc, char* argv[]) {
    utils::InitLogging();

    cli::ArgParser parser("dorado");
    parser.visible.add_description(
            "Alignment using minimap2. The outputs are expected to be equivalent to minimap2.\n"
            "The default parameters use the map-ont preset.\n"
            "NOTE: Not all arguments from minimap2 are currently available. Additionally, "
            "parameter names are not finalized and may change.");
    parser.visible.add_argument("index").help("reference in (fastq/fasta/mmi).");
    parser.visible.add_argument("reads")
            .help("any HTS format.")
            .nargs(argparse::nargs_pattern::any);
    parser.visible.add_argument("-t", "--threads")
            .help("number of threads for alignment and BAM writing.")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("-n", "--max-reads")
            .help("maximum number of reads to process (for debugging, 0=unlimited).")
            .default_value(0)
            .scan<'i', int>();
    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    cli::add_minimap2_arguments(parser, alignment::dflt_options);

    try {
        cli::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.visible.get<bool>("--verbose")) {
        mm_verbose = 3;
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto index(parser.visible.get<std::string>("index"));
    auto reads(parser.visible.get<std::vector<std::string>>("reads"));
    auto threads(parser.visible.get<int>("threads"));
    auto max_reads(parser.visible.get<int>("max-reads"));
    auto options = cli::process_minimap2_arguments(parser, alignment::dflt_options);
    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // alignment. Heuristically use 10% of threads for BAM generation and
    // rest for alignment. Empirically this shows good perf.
    int aligner_threads, writer_threads;
    std::tie(aligner_threads, writer_threads) =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> aligner threads {}, writer threads {}", aligner_threads, writer_threads);

    if (reads.size() == 0) {
#ifndef _WIN32
        if (isatty(fileno(stdin))) {
            std::cout << parser.visible << std::endl;
            return 1;
        }
#endif
        reads.push_back("-");
    } else if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        return 1;
    }

    spdlog::info("> loading index {}", index);

    HtsReader reader(reads[0], std::nullopt);
    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
    auto header = sam_hdr_dup(reader.header);
    add_pg_hdr(header);

    auto output_mode = HtsWriter::OutputMode::BAM;
    if (utils::is_fd_tty(stdout)) {
        output_mode = HtsWriter::OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = HtsWriter::OutputMode::UBAM;
    }

    PipelineDescriptor pipeline_desc;
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, "-", output_mode, writer_threads);
    auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
    auto aligner = pipeline_desc.add_node<AlignerNode>({hts_writer}, index_file_access, index,
                                                       options, aligner_threads);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    const auto& aligner_ref = dynamic_cast<AlignerNode&>(pipeline->get_node_ref(aligner));
    utils::add_sq_hdr(header, aligner_ref.get_sequence_records_for_header());
    auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
    hts_writer_ref.set_and_write_header(header);

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(0, false);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));

    spdlog::info("> starting alignment");
    reader.read(*pipeline, max_reads);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    // Stop the stats sampler thread before tearing down any pipeline objects.
    stats_sampler->terminate();

    tracker.update_progress_bar(final_stats);
    tracker.summarize();

    spdlog::info("> finished alignment");
    spdlog::info("> total/primary/unmapped {}/{}/{}", hts_writer_ref.get_total(),
                 hts_writer_ref.get_primary(), hts_writer_ref.get_unmapped());
    return 0;
}

}  // namespace dorado
