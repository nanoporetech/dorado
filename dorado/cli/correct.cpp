#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "read_pipeline/ErrorCorrectionMapperNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"

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

namespace {

void add_pg_hdr(sam_hdr_t* hdr) {
    sam_hdr_add_line(hdr, "PG", "ID", "correct", "PN", "dorado", "VN", DORADO_VERSION, NULL);
}

}  // anonymous namespace

int correct(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description("Dorado read correction tool.");
    parser.add_argument("reads")
            .help("Path to a file with reads to correct in FASTQ format.")
            .nargs(argparse::nargs_pattern::any);
    parser.add_argument("-t", "--threads")
            .help("Combined number of threads for adapter/primer detection and output generation. "
                  "Default uses "
                  "all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("maximum number of reads to process (for debugging, 0=unlimited).")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to correct.")
            .default_value(std::string(""));
    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(EXIT_FAILURE);
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.get<std::vector<std::string>>("reads"));
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // adapter/primer correctming. Heuristically use 10% of threads for BAM
    // generation and rest for correctming.
    auto [correct_threads, correct_writer_threads] =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> correcting threads {}, writer threads {}", correct_threads,
                  correct_writer_threads);

    auto read_list = utils::load_read_list(parser.get<std::string>("--read-ids"));

    if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        std::exit(EXIT_FAILURE);
    }

    // The overall pipeline will be as follows -
    // 1. The HTSReader will read the fastq input and
    // generate read records.
    // 2. These will go into an Alignment node with the
    // same fastq as an index to perform all-vs-all alignment
    // for reach read. Each alignment is expected to generate
    // multiple aligned records, which will all be packaged
    // and sent into a window generation node.
    // 3. Window generation will chunk up the alignments into
    // multiple windows and create tensors for downstream inference.
    // 4. The inference node will run the model on the tensors, decode
    // the output, and convert each window into its corrected read and stitch
    // adjacent windows together.
    // 5. Corrected reads will be written out FASTA or BAM format.

    // Setup input file reader.
    HtsReader reader(reads[0], read_list);
    auto header = SamHdrPtr(sam_hdr_dup(reader.header));
    add_pg_hdr(header.get());
    utils::strip_alignment_data_from_header(header.get());

    // Setup outut file.
    auto output_mode = OutputMode::BAM;
    utils::HtsFile hts_file("-", output_mode, correct_writer_threads);

    PipelineDescriptor pipeline_desc;
    // 5. Corrected reads will be written out FASTA or BAM format.
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, "");

    // 2. These will go into an Alignment node with the
    // same fastq as an index to perform all-vs-all alignment
    // for reach read. Each alignment is expected to generate
    // multiple aligned records, which will all be packaged
    // and sent into a window generation node.
    auto aligner =
            pipeline_desc.add_node<ErrorCorrectionMapperNode>({hts_writer}, reads[0], threads);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    const auto& aligner_ref =
            dynamic_cast<ErrorCorrectionMapperNode&>(pipeline->get_node_ref(aligner));
    utils::add_sq_hdr(header.get(), aligner_ref.get_sequence_records_for_header());
    hts_file.set_and_write_header(header.get());

    // Set up stats counting.
    ProgressTracker tracker(0, false, hts_file.finalise_is_noop() ? 0.f : 0.5f);
    tracker.set_description("Correcting");
    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
    // End stats counting setup.

    spdlog::info("> starting correction");
    // 1. The HTSReader will read the fastq input and
    // generate read records.
    reader.read(*pipeline, max_reads);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);

    // Report progress during output file finalisation.
    hts_file.finalise([&](size_t) {}, correct_writer_threads, false);
    tracker.summarize();

    spdlog::info("> finished correction");

    return 0;
}

}  // namespace dorado
