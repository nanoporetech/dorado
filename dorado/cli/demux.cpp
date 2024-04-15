#include "alignment/alignment_processing_items.h"
#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/BarcodeDemuxerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
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

BarcodingInfo get_barcoding_info(cli::ArgParser& parser, const utils::SampleSheet* sample_sheet) {
    BarcodingInfo result{};
    result.kit_name = parser.visible.present<std::string>("--kit-name").value_or("");
    result.custom_kit = parser.visible.present<std::string>("--barcode-arrangement");
    if (result.kit_name.empty() && !result.custom_kit) {
        return result;
    }
    result.barcode_both_ends = parser.visible.get<bool>("--barcode-both-ends");
    result.trim = !parser.visible.get<bool>("--no-trim");
    if (sample_sheet) {
        result.allowed_barcodes = sample_sheet->get_barcode_values();
    }
    result.custom_seqs = parser.visible.present("--barcode-sequences");

    return result;
}

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
            .help("Skip barcode trimming. If this option is not chosen, trimming is enabled. "
                  "Note that you should use this option if your input data is mapped and you "
                  "want to preserve the mapping in the output files, as trimming will result "
                  "in any mapping information from the input file(s) being discarded.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--sort-bam")
            .help("Sort any BAM output files that contain mapped reads. Using this option "
                  "requires that the --no-trim option is also set.")
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

    auto no_trim(parser.visible.get<bool>("--no-trim"));
    auto sort_bam(parser.visible.get<bool>("--sort-bam"));
    if (sort_bam && !no_trim) {
        spdlog::error("If --sort-bam is specified then --no-trim must also be specified.");
        std::exit(1);
    }

    auto progress_stats_frequency(parser.hidden.get<int>("progress_stats_frequency"));
    if (progress_stats_frequency > 0) {
        utils::EnsureInfoLoggingEnabled(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    } else {
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
    if (!no_trim) {
        // Remove SQ lines from header since alignment information
        // is invalidated after trimming.
        utils::strip_alignment_data_from_header(header.get());
    }

    auto barcode_sample_sheet = parser.visible.get<std::string>("--sample-sheet");
    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, true);
    }
    BarcodingInfo barcoding_info = get_barcoding_info(parser, sample_sheet.get());

    class DemuxClientInfo : public DefaultClientInfo {
        const BarcodingInfo m_barcoding_info;

    public:
        DemuxClientInfo(BarcodingInfo barcoding_info)
                : m_barcoding_info(std::move(barcoding_info)) {}
        const BarcodingInfo& barcoding_info() const override { return m_barcoding_info; }
    };
    auto client_info = std::make_shared<DemuxClientInfo>(std::move(barcoding_info));
    reader.set_client_info(client_info);

    PipelineDescriptor pipeline_desc;
    auto demux_writer = pipeline_desc.add_node<BarcodeDemuxerNode>(
            {}, output_dir, demux_writer_threads, parser.visible.get<bool>("--emit-fastq"),
            std::move(sample_sheet), sort_bam);

    const auto& info = client_info->barcoding_info();
    if (!info.kit_name.empty() || info.custom_kit) {
        std::vector<std::string> kit_names{};
        if (!info.kit_name.empty()) {
            kit_names.push_back(info.kit_name);
        }
        pipeline_desc.add_node<BarcodeClassifierNode>(
                {demux_writer}, demux_threads, kit_names, info.barcode_both_ends, !info.trim,
                info.allowed_barcodes, info.custom_kit, info.custom_seqs);
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

    // All progress reporting is in the post-processing part.
    ProgressTracker tracker(0, false, 1.f);
    if (progress_stats_frequency > 0) {
        tracker.disable_progress_reporting();
    }
    tracker.set_description("Demuxing");

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });

    ReadOutputProgressStats progress_stats(
            std::chrono::seconds{progress_stats_frequency}, all_files.size(),
            ReadOutputProgressStats::StatsCollectionMode::single_collector);
    progress_stats.set_post_processing_percentage(0.4f);
    progress_stats.start();

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
    progress_stats.notify_stats_collector_completed(final_stats);

    // Finalise the files that were created.
    tracker.set_description("Sorting output files");
    demux_writer_ref.finalise_hts_files([&](size_t progress) {
        tracker.update_post_processing_progress(static_cast<float>(progress));
        progress_stats.update_post_processing_progress(static_cast<float>(progress));
    });

    tracker.summarize();
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
