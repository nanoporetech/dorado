#include "ProgressTracker.h"
#include "alignment/alignment_info.h"
#include "alignment/minimap2_args.h"
#include "basecall_output_args.h"
#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "dorado_version.h"
#include "hts_utils/HeaderMapper.h"
#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "hts_writer/SummaryFileWriter.h"
#include "hts_writer/interface.h"
#include "read_output_progress_stats.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/AlignerNode.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "utils/log_utils.h"
#include "utils/stats.h"
#include "utils/string_utils.h"
#include "utils/tty_utils.h"

#include <minimap.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace std::chrono_literals;

namespace {

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

void add_pg_hdr(sam_hdr_t* hdr) {
    sam_hdr_add_pg(hdr, "aligner", "PN", "dorado", "VN", DORADO_VERSION, "DS", MM_VERSION, nullptr);
}

}  // namespace

namespace dorado {

int aligner(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado aligner", DORADO_VERSION,
                                    argparse::default_arguments::help);
    parser.add_description(
            "Alignment using minimap2. The outputs are expected to be equivalent to minimap2.\n"
            "The default parameters use the lr:hq preset.\n"
            "NOTE: Not all arguments from minimap2 are currently available. Additionally, "
            "parameter names are not finalized and may change.");

    parser.add_argument("index").help("reference in (fastq/fasta/mmi).");
    parser.add_argument("reads")
            .help("An input file or the folder containing input file(s) (any HTS format).")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string{});

    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .action([&](const auto&) { ++verbosity; })
            .append();

    {
        parser.add_group("Input data arguments");
        parser.add_argument("-r", "--recursive")
                .help("If the 'reads' positional argument is a folder any subfolders will also be "
                      "searched for input files.")
                .flag();
        parser.add_argument("-n", "--max-reads")
                .help("maximum number of reads to process (for debugging, 0=unlimited).")
                .default_value(0)
                .scan<'i', int>();
    }
    {
        parser.add_group("Alignment arguments");
        alignment::mm2::add_options_string_arg(parser);
        parser.add_argument("--bed-file")
                .help("Optional bed-file. If specified, overlaps between the alignments and "
                      "bed-file entries will be counted, and recorded in BAM output using the 'bh' "
                      "read tag.")
                .default_value(std::string(""));
    }
    {
        parser.add_group("Output arguments");
        parser.add_argument("--no-sort").help("Disable sorting of output files.").flag();
        cli::add_aligner_output_arguments(parser);
    }
    {
        parser.add_group("Advanced arguments");
        parser.add_argument("-t", "--threads")
                .help("number of threads for alignment and BAM writing (0=unlimited).")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("--allow-sec-supp")
                .help("Align secondary and supplementary records from the input BAM if present.")
                .flag();
    }

    parser.add_argument("--progress_stats_frequency")
            .hidden()
            .help("Frequency in seconds in which to report progress statistics")
            .default_value(0)
            .scan<'i', int>();

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);
    try {
        cli::parse(parser, args_excluding_mm2_opts);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("--verbose")) {
        mm_verbose = 3;
    }

    auto progress_stats_frequency(parser.get<int>("progress_stats_frequency"));
    if (progress_stats_frequency > 0) {
        utils::EnsureInfoLoggingEnabled(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    } else {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }
    auto align_info = std::make_shared<alignment::AlignmentInfo>();
    align_info->reference_file = parser.get<std::string>("index");
    align_info->bed_file = parser.get<std::string>("bed-file");

    const auto reads(parser.get<std::string>("reads"));
    const auto recursive_input = parser.get<bool>("recursive");
    const auto output_dir = cli::get_output_dir(parser);
    const bool skip_sec_supp = !parser.get<bool>("--allow-sec-supp");

    const auto sort_requested = !parser.get<bool>("--no-sort");
    const auto emit_sam = cli::get_emit_sam(parser);
    const auto emit_summary = cli::get_emit_summary(parser);

    auto threads(parser.get<int>("threads"));
    std::size_t max_reads(parser.get<int>("max-reads"));

    std::string err_msg{};
    auto minimap_options = alignment::mm2::try_parse_options(mm2_option_string, err_msg);
    if (!minimap_options) {
        spdlog::error("{}\n{}", err_msg, alignment::mm2::get_help_message());
        return EXIT_FAILURE;
    }
    align_info->minimap_options = std::move(*minimap_options);

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty() && utils::is_fd_tty(stdin)) {
        std::cout << parser << '\n';
        return EXIT_FAILURE;
    }

    if (reads.empty() && output_dir.has_value()) {
        spdlog::error("--output-dir is not valid if input is stdin.");
        return EXIT_FAILURE;
    }

    const auto all_files = cli::collect_inputs(reads, recursive_input);
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

    // All progress reporting is in the post-processing part.
    ProgressTracker tracker(ProgressTracker::Mode::ALIGN, 0, 1.f);
    if (progress_stats_frequency > 0) {
        tracker.disable_progress_reporting();
    }

    bool has_barcoding = false;
    bool strip_input_alignments = true;
    std::unique_ptr<utils::HeaderMapper> header_mapper;
    if (!reads.empty()) {
        header_mapper = std::make_unique<utils::HeaderMapper>(all_files, strip_input_alignments);
        auto hdr = header_mapper->get_shared_merged_header(strip_input_alignments);
        int num_rg_lines = sam_hdr_count_lines(hdr.get(), "RG");
        KString tag_wrapper(100000);
        auto& tag_value = tag_wrapper.get();
        for (int i = 0; i < num_rg_lines; ++i) {
            if (sam_hdr_find_tag_pos(hdr.get(), "RG", i, "SM", &tag_value) == 0) {
                has_barcoding = true;
                break;
            }
        }
    }

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

        auto hts_writer_builder = hts_writer::AlignerHtsFileWriterBuilder(
                emit_sam, sort_requested, output_dir, writer_threads, progress_callback,
                description_callback, has_barcoding);

        std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
        if (hts_file_writer == nullptr) {
            spdlog::error("Failed to create hts file writer");
            std::exit(EXIT_FAILURE);
        }

        writers.push_back(std::move(hts_file_writer));
    }

    hts_writer::SummaryFileWriter::AlignmentCounts alignment_counts;
    hts_writer::SummaryFileWriter::FieldFlags flags;
    if (emit_summary) {
        std::tie(flags, alignment_counts) = cli::make_summary_info(all_files);
        flags |= hts_writer::SummaryFileWriter::ALIGNMENT_FIELDS;
        auto summary_output = output_dir.has_value() ? std::filesystem::path(output_dir.value())
                                                     : std::filesystem::current_path();
        auto summary_writer =
                std::make_unique<hts_writer::SummaryFileWriter>(summary_output, flags);
        writers.push_back(std::move(summary_writer));
    }

    PipelineDescriptor pipeline_desc;
    auto writer_node = pipeline_desc.add_node<WriterNode>({}, std::move(writers));
    auto aligner_node = pipeline_desc.add_node<AlignerNode>(
            {writer_node}, index_file_access, bed_file_access, align_info->reference_file,
            align_info->bed_file, align_info->minimap_options, aligner_threads);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        return EXIT_FAILURE;
    }

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

    tracker.set_description("Collecting Headers");

    // Construct the output headers map
    const auto& aligner_ref = pipeline->get_node_ref<AlignerNode>(aligner_node);
    auto modify_hdr = utils::HeaderMapper::Modifier([&aligner_ref](sam_hdr_t* hdr) {
        add_pg_hdr(hdr);
        utils::add_hd_header_line(hdr);
        utils::add_sq_hdr(hdr, aligner_ref.get_sequence_records_for_header());
    });

    if (header_mapper) {
        if (output_dir.has_value()) {
            header_mapper->modify_headers(modify_hdr);
            // Set the dynamic header map on the writer
            pipeline->get_node_ref<WriterNode>(writer_node)
                    .set_dynamic_header(header_mapper->get_merged_headers_map());
        } else {
            // Convert the dynamic header into a single merged sharable header
            // Strip the alignments and add them back in because merge header finalise
            // can change the tid / sq line indexing
            auto shared_merged_header =
                    header_mapper->get_shared_merged_header(strip_input_alignments);
            modify_hdr(shared_merged_header.get());
            pipeline->get_node_ref<WriterNode>(writer_node)
                    .set_shared_header(std::move(shared_merged_header));
        }
    }

    tracker.set_description("Aligning");
    spdlog::info("> starting alignment");

    for (const auto& file_info : all_files) {
        spdlog::info("processing '{}'", file_info.string());
        HtsReader reader(file_info.string(), std::nullopt);
        if (emit_summary) {
            auto read_initialiser =
                    std::make_shared<hts_writer::SummaryFileWriter::ReadInitialiser>(
                            reader.header(), alignment_counts);
            reader.add_read_initialiser([read_initialiser](HtsData& data) {
                read_initialiser->update_read_attributes(data);
            });
            if (flags & hts_writer::SummaryFileWriter::BARCODING_FIELDS) {
                reader.add_read_initialiser([read_initialiser](HtsData& data) {
                    read_initialiser->update_barcoding_fields(data);
                });
            }
            reader.add_read_initialiser([read_initialiser](HtsData& data) {
                read_initialiser->update_alignment_fields(data);
            });
        }
        reader.set_client_info(client_info);
        spdlog::debug("> input:'{}' fmt:'{}' aligned:'{}'", file_info.filename().string(),
                      reader.format(), reader.is_aligned);

        if (header_mapper == nullptr) {
            SamHdrPtr hdr(sam_hdr_dup(reader.header()));
            modify_hdr(hdr.get());
            pipeline->get_node_ref<WriterNode>(writer_node).set_shared_header(std::move(hdr));
        }

        auto num_reads_in_file = reader.read(*pipeline, max_reads, strip_input_alignments,
                                             header_mapper.get(), skip_sec_supp);
        max_reads -= num_reads_in_file;
        progress_stats.update_reads_per_file_estimate(num_reads_in_file);
    }

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::No});
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);
    tracker.mark_as_completed();
    progress_stats.notify_stats_collector_completed(final_stats);
    progress_stats.report_final_stats();

    spdlog::info("> finished alignment");
    tracker.summarize();

    return EXIT_SUCCESS;
}

}  // namespace dorado
