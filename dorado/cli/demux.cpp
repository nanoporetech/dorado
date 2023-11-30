#include "Version.h"
#include "cli/cli_utils.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/BarcodeDemuxerNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/ProgressTracker.h"
#include "utils/SampleSheet.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"

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

namespace {

void add_pg_hdr(sam_hdr_t* hdr) {
    sam_hdr_add_line(hdr, "PG", "ID", "demux", "PN", "dorado", "VN", DORADO_VERSION, NULL);
}

}  // anonymous namespace

int demuxer(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description("Barcode demultiplexing tool. Users need to specify the kit name(s).");
    parser.add_argument("reads")
            .help("Path to a file with reads to demultiplex. Can be in any HTS format.")
            .nargs(argparse::nargs_pattern::any);
    parser.add_argument("--output-dir").help("Output folder for demultiplexed reads.").required();
    parser.add_argument("--kit-name")
            .help("Barcoding kit name. Cannot be used with --no-classify. Choose "
                  "from: " +
                  dorado::barcode_kits::barcode_kits_list_str() + ".");
    parser.add_argument("--sample-sheet")
            .help("Path to the sample sheet to use.")
            .default_value(std::string(""));
    parser.add_argument("--no-classify")
            .help("Skip barcode classification. Only demux based on existing classification in "
                  "reads. Cannot be used with --kit-name or --sample-sheet.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("-t", "--threads")
            .help("Combined number of threads for barcoding and output generation. Default uses "
                  "all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("Maximum number of reads to process. Mainly for debugging. Process all reads by "
                  "default.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to demux.")
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
    parser.add_argument("--barcode-both-ends")
            .help("Require both ends of a read to be barcoded for a double ended barcode.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--no-trim")
            .help("Skip barcode trimming. If option is not chosen, trimming is enabled.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--barcode-arrangement")
            .help("Path to file with custom barcode arrangement.");
    parser.add_argument("--barcode-sequences").help("Path to file with custom barcode sequences.");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if ((parser.is_used("--no-classify") && parser.is_used("--kit-name")) ||
        (!parser.is_used("--no-classify") && !parser.is_used("--kit-name") &&
         !parser.is_used("--barcode-arrangement"))) {
        spdlog::error(
                "Please specify either --no-classify or --kit-name or pass a custom barcode "
                "arrangement with --barcode-arrangement to use the demux tool.");
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.get<std::vector<std::string>>("reads"));
    auto output_dir(parser.get<std::string>("output-dir"));
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // barcoding. Heuristically use 10% of threads for BAM generation and
    // rest for barcoding. Empirically this shows good perf.
    auto [demux_threads, demux_writer_threads] =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> barcoding threads {}, writer threads {}", demux_threads, demux_writer_threads);

    std::optional<std::string> custom_kit = std::nullopt;
    if (parser.is_used("--barcode-arrangement")) {
        custom_kit = parser.get<std::string>("--barcode-arrangement");
    }

    std::optional<std::string> custom_seqs = std::nullopt;
    if (parser.is_used("--barcode-sequences")) {
        custom_seqs = parser.get<std::string>("--barcode-sequences");
    }

    auto read_list = utils::load_read_list(parser.get<std::string>("--read-ids"));

    if (reads.empty()) {
#ifndef _WIN32
        if (isatty(fileno(stdin))) {
            std::cout << parser << std::endl;
            return 1;
        }
#endif
        reads.push_back("-");
    } else if (reads.size() > 1) {
        spdlog::error("> multi file input not yet handled");
        return 1;
    }

    HtsReader reader(reads[0], read_list);
    auto header = SamHdrPtr(sam_hdr_dup(reader.header));
    add_pg_hdr(header.get());

    auto barcode_sample_sheet = parser.get<std::string>("--sample-sheet");
    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    BarcodingInfo::FilterSet allowed_barcodes;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, true);
        allowed_barcodes = sample_sheet->get_barcode_values();
    }

    PipelineDescriptor pipeline_desc;
    auto demux_writer = pipeline_desc.add_node<BarcodeDemuxerNode>(
            {}, output_dir, demux_writer_threads, parser.get<bool>("--emit-fastq"),
            std::move(sample_sheet));

    if (parser.is_used("--kit-name") || parser.is_used("--barcode-arrangement")) {
        std::vector<std::string> kit_names;
        if (auto names = parser.present<std::vector<std::string>>("--kit-name")) {
            kit_names = std::move(*names);
        }
        pipeline_desc.add_node<BarcodeClassifierNode>(
                {demux_writer}, demux_threads, kit_names, parser.get<bool>("--barcode-both-ends"),
                parser.get<bool>("--no-trim"), std::move(allowed_barcodes), std::move(custom_kit),
                std::move(custom_seqs));
    }

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& demux_writer_ref =
            dynamic_cast<BarcodeDemuxerNode&>(pipeline->get_node_ref(demux_writer));
    demux_writer_ref.set_header(header.get());

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(0, false);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
    // End stats counting setup.

    spdlog::info("> starting barcode demuxing");
    reader.read(*pipeline, max_reads);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    stats_sampler->terminate();

    tracker.update_progress_bar(final_stats);
    tracker.summarize();

    spdlog::info("> finished barcode demuxing");

    return 0;
}

}  // namespace dorado
