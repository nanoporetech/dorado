#include "alignment/alignment_processing_items.h"
#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "demux/barcoding_info.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "dorado_version.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/BarcodeDemuxerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/TrimmerNode.h"
#include "read_pipeline/read_output_progress_stats.h"
#include "summary/summary.h"
#include "utils/MergeHeaders.h"
#include "utils/SampleSheet.h"
#include "utils/arg_parse_ext.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"
#include "utils/tty_utils.h"

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

// This function allows us to map the reference id from input BAM records to what
// they should be in the output file, based on the new ordering of references in
// the merged header.
void adjust_tid(const std::vector<uint32_t>& mapping, dorado::BamPtr& record) {
    auto tid = record.get()->core.tid;
    if (tid >= 0) {
        if (tid >= int32_t(mapping.size())) {
            throw std::range_error("BAM tid field out of range with respect to SQ lines.");
        }
        record.get()->core.tid = int32_t(mapping.at(tid));
    }
}

std::shared_ptr<const dorado::demux::BarcodingInfo> get_barcoding_info(
        dorado::utils::arg_parse::ArgParser& parser,
        const dorado::utils::SampleSheet* sample_sheet) {
    auto result = std::make_shared<dorado::demux::BarcodingInfo>();
    result->kit_name = parser.visible.present<std::string>("--kit-name").value_or("");
    if (result->kit_name.empty()) {
        return nullptr;
    }
    result->barcode_both_ends = parser.visible.get<bool>("--barcode-both-ends");
    result->trim = !parser.visible.get<bool>("--no-trim");
    if (sample_sheet) {
        result->allowed_barcodes = sample_sheet->get_barcode_values();
    }

    return result;
}

}  // anonymous namespace

namespace dorado {

int demuxer(int argc, char* argv[]) {
    utils::arg_parse::ArgParser parser("dorado demux");
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
        utils::arg_parse::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.visible.is_used("--no-classify") == parser.visible.is_used("--kit-name")) {
        spdlog::error("Please specify either --no-classify or --kit-name to use the demux tool.");
        return EXIT_FAILURE;
    }

    auto no_trim(parser.visible.get<bool>("--no-trim") ||
                 parser.visible.get<bool>("--no-classify"));
    auto sort_bam(parser.visible.get<bool>("--sort-bam"));
    if (sort_bam && !no_trim) {
        spdlog::error("If --sort-bam is specified then --no-trim must also be specified.");
        return EXIT_FAILURE;
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
    auto strip_alignment = !no_trim;
    std::vector<std::string> args(argv, argv + argc);

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty() && utils::is_fd_tty(stdin)) {
        std::cout << parser.visible << '\n';
        return EXIT_FAILURE;
    }

    alignment::AlignmentProcessingItems processing_items{reads, recursive_input, output_dir, true};
    if (!processing_items.initialise()) {
        spdlog::error("Could not initialise for input {}", reads);
        return EXIT_FAILURE;
    }
    const auto& all_files = processing_items.get();
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

    auto read_list = utils::load_read_list(parser.visible.get<std::string>("--read-ids"));

    HtsReader reader(all_files[0].input, read_list);
    utils::MergeHeaders hdr_merger(strip_alignment);
    hdr_merger.add_header(reader.header(), all_files[0].input);

    // Fold in the headers from all the other files in the input list.
    for (size_t input_idx = 1; input_idx < all_files.size(); input_idx++) {
        HtsReader header_reader(all_files[input_idx].input, read_list);
        auto error_msg = hdr_merger.add_header(header_reader.header(), all_files[input_idx].input);
        if (!error_msg.empty()) {
            spdlog::error("Unable to combine headers from all input files: " + error_msg);
            return EXIT_FAILURE;
        }
    }

    hdr_merger.finalize_merge();
    auto sq_mapping = hdr_merger.get_sq_mapping();
    auto header = SamHdrPtr(sam_hdr_dup(hdr_merger.get_merged_header()));
    cli::add_pg_hdr(header.get(), "demux", args, "cpu");

    auto barcode_sample_sheet = parser.visible.get<std::string>("--sample-sheet");
    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, true);
    }

    auto client_info = std::make_shared<DefaultClientInfo>();
    reader.set_client_info(client_info);

    auto barcoding_info = get_barcoding_info(parser, sample_sheet.get());

    PipelineDescriptor pipeline_desc;
    auto demux_writer = pipeline_desc.add_node<BarcodeDemuxerNode>(
            {}, output_dir, demux_writer_threads, parser.visible.get<bool>("--emit-fastq"),
            std::move(sample_sheet), sort_bam);

    if (barcoding_info) {
        std::optional<std::string> custom_seqs =
                parser.visible.present<std::string>("--barcode-sequences");
        if (custom_seqs.has_value()) {
            try {
                std::unordered_map<std::string, std::string> custom_barcodes;
                auto custom_sequences = demux::parse_custom_sequences(*custom_seqs);
                for (const auto& entry : custom_sequences) {
                    custom_barcodes.emplace(std::make_pair(entry.name, entry.sequence));
                }
                barcode_kits::add_custom_barcodes(custom_barcodes);
            } catch (const std::exception& e) {
                spdlog::error(e.what());
                std::exit(EXIT_FAILURE);
            } catch (...) {
                spdlog::error("Unable to parse custom sequences file {}", *custom_seqs);
                std::exit(EXIT_FAILURE);
            }
        }

        std::optional<std::string> custom_kit =
                parser.visible.present<std::string>("--barcode-arrangement");
        if (custom_kit.has_value()) {
            try {
                auto [kit_name, kit_info] = demux::get_custom_barcode_kit_info(*custom_kit);
                barcode_kits::add_custom_barcode_kit(kit_name, kit_info);
            } catch (const std::exception& e) {
                spdlog::error("Unable to load custom barcode arrangement file: {}\n{}", *custom_kit,
                              e.what());
                std::exit(EXIT_FAILURE);
            } catch (...) {
                spdlog::error("Unable to load custom barcode arrangement file: {}", *custom_kit);
                std::exit(EXIT_FAILURE);
            }
        }

        if (!barcode_kits::is_valid_barcode_kit(barcoding_info->kit_name)) {
            spdlog::error(
                    "{} is not a valid barcode kit name. Please run the help "
                    "command to find out available barcode kits.",
                    barcoding_info->kit_name);
            std::exit(EXIT_FAILURE);
        }

        client_info->contexts().register_context<const demux::BarcodingInfo>(barcoding_info);
        auto current_node = demux_writer;
        if (!no_trim) {
            current_node = pipeline_desc.add_node<TrimmerNode>({demux_writer}, 1);
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

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& demux_writer_ref = pipeline->get_node_ref<BarcodeDemuxerNode>(demux_writer);
    demux_writer_ref.set_header(header.get());

    // All progress reporting is in the post-processing part.
    ProgressTracker tracker(ProgressTracker::Mode::SIMPLEX, 0, 1.f);
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
    if (!strip_alignment) {
        reader.set_record_mutator(
                [&sq_mapping](BamPtr& record) { adjust_tid(sq_mapping[0], record); });
    }
    auto num_reads_in_file = reader.read(*pipeline, max_reads);
    spdlog::trace("pushed to pipeline: {}", num_reads_in_file);

    progress_stats.update_reads_per_file_estimate(num_reads_in_file);

    // Barcode all the other files passed in
    for (size_t input_idx = 1; input_idx < all_files.size(); input_idx++) {
        HtsReader input_reader(all_files[input_idx].input, read_list);
        input_reader.set_client_info(client_info);
        if (!strip_alignment) {
            input_reader.set_record_mutator([&sq_mapping, input_idx](BamPtr& record) {
                adjust_tid(sq_mapping[input_idx], record);
            });
        }
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

    return EXIT_SUCCESS;
}

}  // namespace dorado
