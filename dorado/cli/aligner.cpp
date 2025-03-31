#include "alignment/IndexFileAccess.h"
#include "alignment/alignment_info.h"
#include "alignment/alignment_processing_items.h"
#include "alignment/minimap2_args.h"
#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/read_output_progress_stats.h"
#include "summary/summary.h"
#include "utils/PostCondition.h"
#include "utils/arg_parse_ext.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"
#include "utils/tty_utils.h"

#include <minimap.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace std::chrono_literals;

namespace {

constexpr size_t BAM_BUFFER_SIZE = 1000000000;  // 1 GB

std::shared_ptr<dorado::alignment::IndexFileAccess> load_index(
        const std::string& filename,
        const dorado::alignment::Minimap2Options& options,
        const int num_threads) {
    spdlog::info("> loading index {}", filename);

    auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
    int num_index_construction_threads{
            dorado::alignment::mm2::print_aln_seq() ? 1 : static_cast<int>(num_threads)};
    switch (index_file_access->load_index(filename, options, num_index_construction_threads)) {
    case dorado::alignment::IndexLoadResult::reference_file_not_found:
        throw std::runtime_error("Alignment reference path does not exist: " + filename);
    case dorado::alignment::IndexLoadResult::validation_error:
        throw std::runtime_error("Validation error checking minimap options");
    case dorado::alignment::IndexLoadResult::file_open_error:
        throw std::runtime_error("Error opening index file: " + filename);
    case dorado::alignment::IndexLoadResult::no_index_loaded:
    case dorado::alignment::IndexLoadResult::end_of_index:
        throw std::runtime_error(
                "dorado aligner - index loading error. Should not reach this condition.");
    case dorado::alignment::IndexLoadResult::success:
        break;
    }
    return index_file_access;
}

std::shared_ptr<dorado::alignment::BedFileAccess> load_bed(const std::string& filename) {
    auto bed_file_access = std::make_shared<dorado::alignment::BedFileAccess>();
    if (!filename.empty()) {
        if (!bed_file_access->load_bedfile(filename)) {
            throw std::runtime_error("AlignerNode bed-file could not be loaded: " + filename);
        }
    }
    return bed_file_access;
}

bool create_output_folder(const std::filesystem::path& output_folder) {
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    std::filesystem::create_directories(output_folder, creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder {}. ErrorCode({}) {}", output_folder.string(),
                      creation_error.value(), creation_error.message());
        return false;
    }
    return true;
}

void add_pg_hdr(sam_hdr_t* hdr) {
    sam_hdr_add_pg(hdr, "aligner", "PN", "dorado", "VN", DORADO_VERSION, "DS", MM_VERSION, nullptr);
}

}  // namespace

namespace dorado {

int aligner(int argc, char* argv[]) {
    utils::arg_parse::ArgParser parser("dorado aligner");
    parser.visible.add_description(
            "Alignment using minimap2. The outputs are expected to be equivalent to minimap2.\n"
            "The default parameters use the lr:hq preset.\n"
            "NOTE: Not all arguments from minimap2 are currently available. Additionally, "
            "parameter names are not finalized and may change.");
    parser.visible.add_argument("index").help("reference in (fastq/fasta/mmi).");
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
            .help("If specified output files will be written to the given folder, otherwise output "
                  "is to stdout. Required if the 'reads' positional argument is a folder.")
            .default_value(std::string{});
    parser.visible.add_argument("--emit-summary")
            .help("If specified, a summary file containing the details of the primary alignments "
                  "for each read will be emitted to the root of the output folder. This option "
                  "requires that the '--output-dir' option is also set.")
            .default_value(false)
            .implicit_value(true)
            .nargs(0);
    parser.visible.add_argument("--bed-file")
            .help("Optional bed-file. If specified, overlaps between the alignments and bed-file "
                  "entries will be counted, and recorded in BAM output using the 'bh' read tag.")
            .default_value(std::string(""));
    parser.hidden.add_argument("--progress_stats_frequency")
            .help("Frequency in seconds in which to report progress statistics")
            .default_value(0)
            .scan<'i', int>();
    parser.visible.add_argument("-t", "--threads")
            .help("number of threads for alignment and BAM writing (0=unlimited).")
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

    alignment::mm2::add_options_string_arg(parser);

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);
    try {
        utils::arg_parse::parse(parser, args_excluding_mm2_opts);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.visible.get<bool>("--verbose")) {
        mm_verbose = 3;
    }

    auto progress_stats_frequency(parser.hidden.get<int>("progress_stats_frequency"));
    if (progress_stats_frequency > 0) {
        utils::EnsureInfoLoggingEnabled(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    } else {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }
    auto align_info = std::make_shared<alignment::AlignmentInfo>();
    align_info->reference_file = parser.visible.get<std::string>("index");
    align_info->bed_file = parser.visible.get<std::string>("bed-file");
    auto reads(parser.visible.get<std::string>("reads"));
    auto recursive_input = parser.visible.get<bool>("recursive");
    auto output_folder = parser.visible.get<std::string>("output-dir");

    auto emit_summary = parser.visible.get<bool>("emit-summary");
    if (emit_summary && output_folder.empty()) {
        spdlog::error("Cannot specify '--emit-summary' if '--output-dir' is not also specified.");
        return EXIT_FAILURE;
    }

    auto threads(parser.visible.get<int>("threads"));

    auto max_reads(parser.visible.get<int>("max-reads"));

    std::string err_msg{};
    auto minimap_options = alignment::mm2::try_parse_options(mm2_option_string, err_msg);
    if (!minimap_options) {
        spdlog::error("{}\n{}", err_msg, alignment::mm2::get_help_message());
        return EXIT_FAILURE;
    }
    align_info->minimap_options = std::move(*minimap_options);

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty() && utils::is_fd_tty(stdin)) {
        std::cout << parser.visible << '\n';
        return EXIT_FAILURE;
    }

    alignment::AlignmentProcessingItems processing_items{reads, recursive_input, output_folder,
                                                         false};
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
    // alignment. Heuristically use 10% of threads for BAM generation and
    // rest for alignment. Empirically this shows good perf.
    int aligner_threads, writer_threads;
    std::tie(aligner_threads, writer_threads) =
            cli::worker_vs_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> aligner threads {}, writer threads {}", aligner_threads, writer_threads);

    std::shared_ptr<dorado::alignment::IndexFileAccess> index_file_access;
    try {
        index_file_access = load_index(align_info->reference_file, align_info->minimap_options,
                                       aligner_threads);
    } catch (const std::exception& e) {
        spdlog::error("Index file loading failed: {}", e.what());
        return EXIT_FAILURE;
    }
    std::shared_ptr<dorado::alignment::BedFileAccess> bed_file_access;
    try {
        bed_file_access = load_bed(align_info->bed_file);
    } catch (const std::exception& e) {
        spdlog::error("bed file loading failed: {}", e.what());
        return EXIT_FAILURE;
    }

    ReadOutputProgressStats progress_stats(
            std::chrono::seconds{progress_stats_frequency}, all_files.size(),
            ReadOutputProgressStats::StatsCollectionMode::collector_per_input_file);
    progress_stats.set_post_processing_percentage(0.5f);
    progress_stats.start();

    auto client_info = std::make_shared<DefaultClientInfo>();
    client_info->contexts().register_context<const alignment::AlignmentInfo>(align_info);

    for (const auto& file_info : all_files) {
        spdlog::info("processing {} -> {}", file_info.input, file_info.output);
        auto reader = std::make_unique<HtsReader>(file_info.input, std::nullopt);
        reader->set_client_info(client_info);
        if (file_info.output != "-" &&
            !create_output_folder(std::filesystem::path(file_info.output).parent_path())) {
            return EXIT_FAILURE;
        }

        spdlog::debug("> input fmt: {} aligned: {}", reader->format(), reader->is_aligned);
        auto header = SamHdrPtr(sam_hdr_dup(reader->header()));
        utils::add_hd_header_line(header.get());
        add_pg_hdr(header.get());
        dorado::utils::strip_alignment_data_from_header(header.get());

        const bool sort_bam = (file_info.output_mode == utils::HtsFile::OutputMode::BAM &&
                               file_info.output != "-");
        utils::HtsFile hts_file(file_info.output, file_info.output_mode, writer_threads, sort_bam);
        if (sort_bam) {
            hts_file.set_buffer_size(BAM_BUFFER_SIZE);
        }
        PipelineDescriptor pipeline_desc;
        auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, "");
        auto aligner = pipeline_desc.add_node<AlignerNode>(
                {hts_writer}, index_file_access, bed_file_access, align_info->reference_file,
                align_info->bed_file, align_info->minimap_options, aligner_threads);

        // Create the Pipeline from our description.
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
        if (pipeline == nullptr) {
            spdlog::error("Failed to create pipeline");
            return EXIT_FAILURE;
        }

        // At present, header output file header writing relies on direct node method calls
        // rather than the pipeline framework.
        const auto& aligner_ref = pipeline->get_node_ref<AlignerNode>(aligner);
        utils::add_sq_hdr(header.get(), aligner_ref.get_sequence_records_for_header());
        auto& hts_writer_ref = pipeline->get_node_ref<HtsWriter>(hts_writer);
        hts_file.set_header(header.get());

        // All progress reporting is in the post-processing part.
        ProgressTracker tracker(ProgressTracker::Mode::ALIGN, 0, 1.f);
        if (progress_stats_frequency > 0) {
            tracker.disable_progress_reporting();
        }
        tracker.set_description("Aligning");

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

        spdlog::info("> starting alignment");
        auto num_reads_in_file = reader->read(*pipeline, max_reads);

        // Wait for the pipeline to complete.  When it does, we collect
        // final stats to allow accurate summarisation.
        auto final_stats = pipeline->terminate(DefaultFlushOptions());

        // Stop the stats sampler thread before tearing down any pipeline objects.
        stats_sampler->terminate();
        tracker.update_progress_bar(final_stats);
        progress_stats.update_reads_per_file_estimate(num_reads_in_file);
        progress_stats.notify_stats_collector_completed(final_stats);

        spdlog::info("> finished alignment");

        // Report progress during output file finalisation.
        if (!hts_file.finalise_is_noop()) {
            spdlog::info("> merging temporary BAM files");
        }
        tracker.set_description("Merging temporary BAM files");
        hts_file.finalise([&](size_t progress) {
            tracker.update_post_processing_progress(static_cast<float>(progress));
            progress_stats.update_post_processing_progress(static_cast<float>(progress));
        });

        progress_stats.notify_post_processing_completed();
        tracker.summarize();

        spdlog::info("> total/primary/unmapped {}/{}/{}", hts_writer_ref.get_total(),
                     hts_writer_ref.get_primary(), hts_writer_ref.get_unmapped());
    }

    progress_stats.report_final_stats();

    if (emit_summary) {
        spdlog::info("> generating summary file");
        SummaryData summary(SummaryData::ALIGNMENT_FIELDS);
        auto summary_file = std::filesystem::path(output_folder) / "alignment_summary.txt";
        std::ofstream summary_out(summary_file.string());
        summary.process_tree(output_folder, summary_out);
        spdlog::info("> summary file complete.");
    }

    return EXIT_SUCCESS;
}

}  // namespace dorado
