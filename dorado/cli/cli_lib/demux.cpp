#include "ProgressTracker.h"
#include "alignment/alignment_processing_items.h"
#include "basecall_output_args.h"
#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "demux/barcoding_info.h"
#include "demux/parse_custom_kit.h"
#include "dorado_version.h"
#include "hts_utils/HeaderMapper.h"
#include "hts_utils/bam_utils.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "hts_writer/interface.h"
#include "read_output_progress_stats.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/BarcodeClassifierNode.h"
#include "read_pipeline/nodes/TrimmerNode.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "summary/summary.h"
#include "utils/SampleSheet.h"
#include "utils/arg_parse_ext.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"
#include "utils/tty_utils.h"

#include <argparse/argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

std::shared_ptr<const dorado::demux::BarcodingInfo> get_barcoding_info(
        argparse::ArgumentParser& parser) {
    auto result = std::make_shared<dorado::demux::BarcodingInfo>();
    result->kit_name = parser.present<std::string>("--kit-name").value_or("");
    if (result->kit_name.empty()) {
        return nullptr;
    }
    result->barcode_both_ends = parser.get<bool>("--barcode-both-ends");
    result->trim = !parser.get<bool>("--no-trim");
    auto barcode_sample_sheet = parser.get<std::string>("--sample-sheet");
    if (!barcode_sample_sheet.empty()) {
        result->sample_sheet =
                std::make_shared<const dorado::utils::SampleSheet>(barcode_sample_sheet, true);
        result->allowed_barcodes = result->sample_sheet->get_barcode_values();
    }

    return result;
}

}  // anonymous namespace

namespace dorado {

int demuxer(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado demux", DORADO_VERSION,
                                    argparse::default_arguments::help);
    parser.add_description("Barcode demultiplexing tool. Users need to specify the kit name(s).");
    parser.add_argument("reads")
            .help("An input file or the folder containing input file(s) (any HTS format).")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string{});

    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    {
        parser.add_group("Input data arguments");
        parser.add_argument("-r", "--recursive")
                .help("If the 'reads' positional argument is a folder any subfolders will also be "
                      "searched for input files.")
                .flag();
        parser.add_argument("-n", "--max-reads")
                .help("Maximum number of reads to process. Mainly for debugging. Process all reads "
                      "by default.")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("-l", "--read-ids")
                .help("A file with a newline-delimited list of reads to demux.")
                .default_value(std::string(""));
    }
    {
        parser.add_group("Output arguments");
        cli::add_demux_output_arguments(parser);
        parser.add_argument("--sort-bam")
                .help("Sort any BAM output files that contain mapped reads. Using this option "
                      "requires that the --no-trim option is also set.")
                .flag();
    }
    {
        parser.add_group("Barcoding arguments");
        parser.add_argument("--no-classify")
                .help("Skip barcode classification. Only demux based on existing classification in "
                      "reads. Cannot be used with --kit-name or --sample-sheet.")
                .flag();
        parser.add_argument("--kit-name")
                .help("Barcoding kit name. Cannot be used with --no-classify. Choose "
                      "from: " +
                      dorado::barcode_kits::barcode_kits_list_str() + ".");
        parser.add_argument("--sample-sheet")
                .help("Path to the sample sheet to use.")
                .default_value(std::string(""));
        parser.add_argument("--barcode-both-ends")
                .help("Require both ends of a read to be barcoded for a double ended barcode.")
                .flag();
        parser.add_argument("--barcode-arrangement")
                .help("Path to file with custom barcode arrangement.");
        parser.add_argument("--barcode-sequences")
                .help("Path to file with custom barcode sequences.");
    }
    {
        parser.add_group("Trimming arguments");
        parser.add_argument("--no-trim")
                .help("Skip barcode trimming. If this option is not chosen, trimming is enabled. "
                      "Note that you should use this option if your input data is mapped and you "
                      "want to preserve the mapping in the output files, as trimming will result "
                      "in any mapping information from the input file(s) being discarded.")
                .flag();
    }
    {
        parser.add_group("Advanced arguments");
        parser.add_argument("-t", "--threads")
                .help("Combined number of threads for barcoding and output generation. Default "
                      "uses all available threads.")
                .default_value(0)
                .scan<'i', int>();
    }
    parser.add_argument("--progress_stats_frequency")
            .hidden()
            .help("Frequency in seconds in which to report progress statistics")
            .default_value(0)
            .scan<'i', int>();

    try {
        cli::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.is_used("--no-classify") == parser.is_used("--kit-name")) {
        spdlog::error("Please specify either --no-classify or --kit-name to use the demux tool.");
        return EXIT_FAILURE;
    }

    auto no_trim(parser.get<bool>("--no-trim") || parser.get<bool>("--no-classify"));
    auto sort_bam(parser.get<bool>("--sort-bam"));
    if (sort_bam && !no_trim) {
        spdlog::error("If --sort-bam is specified then --no-trim must also be specified.");
        return EXIT_FAILURE;
    }

    auto progress_stats_frequency(parser.get<int>("progress_stats_frequency"));
    if (progress_stats_frequency > 0) {
        utils::EnsureInfoLoggingEnabled(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    } else {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    const std::string reads(parser.get<std::string>("reads"));
    const std::string output_dir = cli::get_output_dir(parser).value_or(".");
    const bool recursive_input(parser.get<bool>("recursive"));
    const bool emit_fastq = cli::get_emit_fastq(parser);
    const bool emit_summary = cli::get_emit_summary(parser);
    int threads(parser.get<int>("threads"));
    std::size_t max_reads(parser.get<int>("max-reads"));

    auto strip_alignment = !no_trim;
    std::vector<std::string> args(argv, argv + argc);

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty() && utils::is_fd_tty(stdin)) {
        std::cout << parser << '\n';
        return EXIT_FAILURE;
    }

    const auto all_files = alignment::collect_inputs(reads, recursive_input);
    if (all_files.empty()) {
        spdlog::info("No input files found");
        return EXIT_SUCCESS;
    }
    spdlog::info("num input files: {}", all_files.size());

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // barcoding. Heuristically use 10% of threads for BAM generation and
    // rest for barcoding. Empirically this shows good perf.
    auto [demux_threads, demux_writer_threads] =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> barcoding threads {}, writer threads {}", demux_threads, demux_writer_threads);

    auto read_list = utils::load_read_list(parser.get<std::string>("--read-ids"));

    auto barcoding_info = get_barcoding_info(parser);

    // All progress reporting is in the post-processing part.
    ProgressTracker tracker(ProgressTracker::Mode::DEMUX, 0, 1.f);
    if (progress_stats_frequency > 0) {
        tracker.disable_progress_reporting();
    }
    tracker.set_description("Demuxing");

    ReadOutputProgressStats progress_stats(
            std::chrono::seconds{progress_stats_frequency}, all_files.size(),
            ReadOutputProgressStats::StatsCollectionMode::single_collector);
    progress_stats.set_post_processing_percentage(0.4f);
    progress_stats.start();

    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        auto progress_callback =
                utils::ProgressCallback([&tracker, &progress_stats](size_t progress) {
                    // Called as part of hts finalize
                    tracker.update_post_processing_progress(static_cast<float>(progress));
                    progress_stats.update_post_processing_progress(static_cast<float>(progress));
                });
        auto description_callback =
                utils::DescriptionCallback([&tracker](const std::string& description) {
                    tracker.set_description(description);
                });

        auto hts_writer_builder = hts_writer::DemuxHtsFileWriterBuilder(
                emit_fastq, sort_bam, output_dir, demux_writer_threads, progress_callback,
                description_callback, "");

        std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
        if (hts_file_writer == nullptr) {
            spdlog::error("Failed to create hts file writer");
            std::exit(EXIT_FAILURE);
        }

        writers.push_back(std::move(hts_file_writer));
    }

    auto client_info = std::make_shared<DefaultClientInfo>();

    PipelineDescriptor pipeline_desc;
    auto writer_node = pipeline_desc.add_node<WriterNode>({}, std::move(writers));

    if (barcoding_info) {
        if (!demux::try_configure_custom_barcode_sequences(
                    parser.present<std::string>("--barcode-sequences"))) {
            std::exit(EXIT_FAILURE);
        }

        if (!demux::try_configure_custom_barcode_arrangement(
                    parser.present<std::string>("--barcode-arrangement"))) {
            std::exit(EXIT_FAILURE);
        }

        if (!barcode_kits::is_valid_barcode_kit(barcoding_info->kit_name)) {
            spdlog::error(
                    "{} is not a valid barcode kit name. Please run the help "
                    "command to find out available barcode kits.",
                    barcoding_info->kit_name);
            std::exit(EXIT_FAILURE);
        }

        client_info->contexts().register_context<const demux::BarcodingInfo>(barcoding_info);
        auto current_node = writer_node;
        if (!no_trim) {
            current_node = pipeline_desc.add_node<TrimmerNode>({writer_node}, 1);
        }
        pipeline_desc.add_node<BarcodeClassifierNode>({current_node}, demux_threads);
    }

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        return EXIT_FAILURE;
    }

    auto header_mapper = utils::HeaderMapper(all_files, strip_alignment);
    auto add_pg_hdr = utils::HeaderMapper::Modifier(
            [&args](sam_hdr_t* hdr) { cli::add_pg_hdr(hdr, "demux", args, "cpu"); });
    auto update_barcode_rg_groups = utils::HeaderMapper::Modifier([&](sam_hdr_t* hdr) {
        if (!barcoding_info) {
            return;
        }

        auto found_read_groups = utils::parse_read_groups(hdr);
        utils::add_rg_headers_with_barcode_kit(hdr, found_read_groups, barcoding_info->kit_name,
                                               barcoding_info->sample_sheet.get());
    });
    header_mapper.modify_headers(add_pg_hdr);
    header_mapper.modify_headers(update_barcode_rg_groups);

    // Set the dynamic header map
    pipeline->get_node_ref<WriterNode>(writer_node)
            .set_dynamic_header(header_mapper.get_merged_headers_map());

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });

    stats_callables.push_back([&progress_stats](const stats::NamedStats& stats) {
        progress_stats.update_stats(stats);
    });

    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, static_cast<size_t>(0));
    // End stats counting setup.

    spdlog::info("> starting barcode demuxing");
    for (const auto& input : all_files) {
        HtsReader reader(input.string(), read_list);
        reader.set_client_info(client_info);

        const auto num_reads_in_file =
                reader.read(*pipeline, max_reads, strip_alignment, &header_mapper, false);
        max_reads -= num_reads_in_file;
        spdlog::trace("pushed to pipeline: {}", num_reads_in_file);
        progress_stats.update_reads_per_file_estimate(num_reads_in_file);
    }

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate({.fast = dorado::utils::AsyncQueueTerminateFast::No});
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);
    progress_stats.notify_stats_collector_completed(final_stats);

    // Finalise the files that were created.
    tracker.set_description("Sorting output files");

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

    return EXIT_SUCCESS;
}

}  // namespace dorado
