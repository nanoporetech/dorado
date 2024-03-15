#include "Version.h"
#include "alignment/alignment_processing_items.h"
#include "cli/cli_utils.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/BarcodeDemuxerNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/read_output_progress_stats.h"
#include "summary/summary.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
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
    cli::ArgParser parser("dorado demux");
    parser.visible.add_description(
            "Barcode demultiplexing tool. Users need to specify the kit name(s).");
    parser.visible.add_argument("reads")
            .help("An input file or the folder containing input file(s) (any HTS format).")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string{});
    parser.visible.add_argument("-r", "--recursive")
            .help("If the 'reads' positional argument is a folder any subfolders will also be "
                  "searched for input files.")
            .default_value(false)
            .implicit_value(true)
            .nargs(0);
    parser.visible.add_argument("-o", "--output-dir")
            .help("Output folder for demultiplexed reads.")
            .required();
    parser.visible.add_argument("--kit-name")
            .help("Barcoding kit name. Cannot be used with --no-classify. Choose "
                  "from: " +
                  dorado::barcode_kits::barcode_kits_list_str() + ".");
    parser.visible.add_argument("--sample-sheet")
            .help("Path to the sample sheet to use.")
            .default_value(std::string(""));
    parser.visible.add_argument("--no-classify")
            .help("Skip barcode classification. Only demux based on existing classification in "
                  "reads. Cannot be used with --kit-name or --sample-sheet.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("-t", "--threads")
            .help("Combined number of threads for barcoding and output generation. Default uses "
                  "all available threads.")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("-n", "--max-reads")
            .help("Maximum number of reads to process. Mainly for debugging. Process all reads by "
                  "default.")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to demux.")
            .default_value(std::string(""));
    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();
    parser.visible.add_argument("--emit-fastq")
            .help("Output in fastq format. Default is BAM.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--emit-summary")
            .help("If specified, a summary file containing the details of the primary alignments "
                  "for each "
                  "read will be emitted to the root of the output folder.")
            .default_value(false)
            .implicit_value(true)
            .nargs(0);
    parser.hidden.add_argument("--progress_stats_frequency")
            .help("Frequency in seconds in which to report progress statistics")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("--barcode-both-ends")
            .help("Require both ends of a read to be barcoded for a double ended barcode.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--no-trim")
            .help("Skip barcode trimming. If option is not chosen, trimming is enabled.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--barcode-arrangement")
            .help("Path to file with custom barcode arrangement.");
    parser.visible.add_argument("--barcode-sequences")
            .help("Path to file with custom barcode sequences.");

    try {
        cli::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if ((parser.visible.is_used("--no-classify") && parser.visible.is_used("--kit-name")) ||
        (!parser.visible.is_used("--no-classify") && !parser.visible.is_used("--kit-name") &&
         !parser.visible.is_used("--barcode-arrangement"))) {
        spdlog::error(
                "Please specify either --no-classify or --kit-name or pass a custom barcode "
                "arrangement with --barcode-arrangement to use the demux tool.");
        std::exit(1);
    }

    if (parser.visible.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.visible.get<std::string>("reads"));
    auto recursive_input(parser.visible.get<bool>("recursive"));
    auto output_dir(parser.visible.get<std::string>("output-dir"));
    auto emit_summary = parser.visible.get<bool>("emit-summary");
    auto threads(parser.visible.get<int>("threads"));
    auto max_reads(parser.visible.get<int>("max-reads"));

    alignment::AlignmentProcessingItems processing_items{reads, recursive_input, output_dir, true};
    if (!processing_items.initialise()) {
        return EXIT_FAILURE;
    }
    const auto& all_files = processing_items.get();
    spdlog::info("num input files: {}", all_files.size());

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // barcoding. Heuristically use 10% of threads for BAM generation and
    // rest for barcoding. Empirically this shows good perf.
    auto [demux_threads, demux_writer_threads] =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> barcoding threads {}, writer threads {}", demux_threads, demux_writer_threads);

    std::optional<std::string> custom_kit = std::nullopt;
    if (parser.visible.is_used("--barcode-arrangement")) {
        custom_kit = parser.visible.get<std::string>("--barcode-arrangement");
    }

    std::optional<std::string> custom_barcode_file = std::nullopt;
    if (parser.visible.is_used("--barcode-sequences")) {
        custom_barcode_file = parser.visible.get<std::string>("--barcode-sequences");
    }

    auto read_list = utils::load_read_list(parser.visible.get<std::string>("--read-ids"));

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty()) {
#ifndef _WIN32
        if (isatty(fileno(stdin))) {
            std::cout << parser.visible << std::endl;
            return 1;
        }
#endif
    }

    HtsReader reader(all_files[0].input, read_list);
    auto header = SamHdrPtr(sam_hdr_dup(reader.header));

    // Fold in the headers from all the other files in the input list.
    for (size_t input_idx = 1; input_idx < all_files.size(); input_idx++) {
        HtsReader header_reader(all_files[input_idx].input, read_list);
        std::string error_msg;
        if (!utils::sam_hdr_merge(header.get(), header_reader.header, error_msg)) {
            spdlog::error("Unable to combine headers from all input files: " + error_msg);
            std::exit(EXIT_FAILURE);
        }
    }

    add_pg_hdr(header.get());

    auto barcode_sample_sheet = parser.visible.get<std::string>("--sample-sheet");
    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    BarcodingInfo::FilterSet allowed_barcodes;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, true);
        allowed_barcodes = sample_sheet->get_barcode_values();
    }

    PipelineDescriptor pipeline_desc;
    auto demux_writer = pipeline_desc.add_node<BarcodeDemuxerNode>(
            {}, output_dir, demux_writer_threads, parser.visible.get<bool>("--emit-fastq"),
            std::move(sample_sheet));

    if (parser.visible.is_used("--kit-name") || parser.visible.is_used("--barcode-arrangement")) {
        std::vector<std::string> kit_names;
        if (auto names = parser.visible.present<std::vector<std::string>>("--kit-name")) {
            kit_names = std::move(*names);
        }
        pipeline_desc.add_node<BarcodeClassifierNode>(
                {demux_writer}, demux_threads, kit_names,
                parser.visible.get<bool>("--barcode-both-ends"),
                parser.visible.get<bool>("--no-trim"), std::move(allowed_barcodes),
                std::move(custom_kit), std::move(custom_barcode_file));
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

    auto progress_stats_frequency(parser.hidden.get<int>("progress_stats_frequency"));
    ReadOutputProgressStats progress_stats(
            std::chrono::seconds{progress_stats_frequency}, all_files.size(),
            ReadOutputProgressStats::StatsCollectionMode::single_collector);
    stats_callables.push_back([&progress_stats](const stats::NamedStats& stats) {
        progress_stats.update_stats(stats);
    });

    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));

    // End stats counting setup.

    spdlog::info("> starting barcode demuxing");
    auto num_reads_in_file = reader.read(*pipeline, max_reads);
    spdlog::trace("pushed to pipeline: {}", num_reads_in_file);

    progress_stats.update_reads_per_file_estimate(num_reads_in_file);

    // Barcode all the other files passed in
    for (size_t input_idx = 1; input_idx < all_files.size(); input_idx++) {
        HtsReader input_reader(all_files[input_idx].input, read_list);
        num_reads_in_file = input_reader.read(*pipeline, max_reads);
        spdlog::trace("pushed to pipeline: {}", num_reads_in_file);
        progress_stats.update_reads_per_file_estimate(num_reads_in_file);
    }

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    stats_sampler->terminate();

    tracker.update_progress_bar(final_stats);
    tracker.summarize();
    progress_stats.notify_stats_collector_completed(final_stats);
    progress_stats.report_final_stats();

    spdlog::info("> finished barcode demuxing");
    if (emit_summary) {
        spdlog::info("> generating summary file");
        SummaryData summary(SummaryData::BARCODING_FIELDS);
        auto summary_file = std::filesystem::path(output_dir) / "barcoding_summary.txt";
        std::ofstream summary_out(summary_file.string());
        summary.process_tree(output_dir, summary_out);
        spdlog::info("> summary file complete.");
    }

    return 0;
}

}  // namespace dorado
