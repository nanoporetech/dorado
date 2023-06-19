#include "Version.h"
#include "minimap.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "utils/bam_utils.h"
#include "utils/cli_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

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

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description(
            "Alignment using minimap2. The outputs are expected to be equivalent to minimap2.\n"
            "The default parameters use the map-ont preset.\n"
            "NOTE: Not all arguments from minimap2 are currently available. Additionally, "
            "parameter names are not finalized and may change.");
    parser.add_argument("index").help("reference in (fastq/fasta/mmi).");
    parser.add_argument("reads").help("any HTS format.").nargs(argparse::nargs_pattern::any);
    parser.add_argument("-t", "--threads")
            .help("number of threads for alignment and BAM writing.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("maxium number of reads to process (for debugging).")
            .default_value(1000000)
            .scan<'i', int>();
    parser.add_argument("-k").help("k-mer size (maximum 28).").default_value(15).scan<'i', int>();
    parser.add_argument("-w").help("minimizer window size.").default_value(10).scan<'i', int>();
    parser.add_argument("-I").help("minimap2 index batch size.").default_value(std::string("16G"));
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        mm_verbose = 3;
        spdlog::set_level(spdlog::level::debug);
    }

    auto index(parser.get<std::string>("index"));
    auto reads(parser.get<std::vector<std::string>>("reads"));
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));
    auto kmer_size(parser.get<int>("k"));
    auto window_size(parser.get<int>("w"));
    auto index_batch_size = utils::parse_string_to_size(parser.get<std::string>("I"));

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // alignment. Heuristically use 10% of threads for BAM generation and
    // rest for alignment. Empirically this shows good perf.
    int aligner_threads, writer_threads;
    std::tie(aligner_threads, writer_threads) =
            utils::aligner_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> aligner threads {}, writer threads {}", aligner_threads, writer_threads);

    if (reads.size() == 0) {
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

    spdlog::info("> loading index {}", index);

    HtsReader reader(reads[0]);
    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
    auto header = sam_hdr_dup(reader.header);
    add_pg_hdr(header);

    PipelineDescriptor pipeline_desc;
    sq_t header_sequence_records;
    auto aligner = pipeline_desc.add_node<Aligner>({}, index, kmer_size, window_size, index_batch_size, aligner_threads, header_sequence_records);
    utils::add_sq_hdr(header, header_sequence_records);
    auto writer = pipeline_desc.add_node<HtsWriter>({}, "-", HtsWriter::OutputMode::BAM, writer_threads, 0, header);
    pipeline_desc.add_node_sink(aligner, writer);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(0, false);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(kStatsPeriod, stats_reporters,
                                                                  stats_callables);

    spdlog::info("> starting alignment");
    reader.read(*pipeline, max_reads);

    pipeline->wait_until_done();

    stats_sampler->terminate();
    tracker.summarize();

    spdlog::info("> finished alignment");

    return 0;
}

}  // namespace dorado
