#include "read_pipeline/Barcoder.h"

#include "Version.h"
#include "read_pipeline/BarcodeDemuxer.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "utils/cli_utils.h"
#include "utils/log_utils.h"
#include "utils/stats.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>
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
    sam_hdr_add_line(hdr, "PG", "ID", "barcoder", "PN", "dorado", "VN", DORADO_VERSION, "DS",
                     MM_VERSION, NULL);
}

}  // anonymous namespace

int barcoder(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description("Barcoding tool. Users need to pass the kit name.");
    parser.add_argument("reads").help("any HTS format.").nargs(argparse::nargs_pattern::any);
    parser.add_argument("--output-dir").help("Output folder for demuxed reads.").required();
    parser.add_argument("-t", "--threads")
            .help("number of threads for barcoding and BAM writing.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("maxium number of reads to process (for debugging).")
            .default_value(10000000)
            .scan<'i', int>();
    parser.add_argument("--kit_name").help("kit name");
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
        utils::SetDebugLogging();
    }

    auto reads(parser.get<std::vector<std::string>>("reads"));
    auto output_dir(parser.get<std::string>("output-dir"));
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    // The input thread is the total number of threads to use for dorado
    // barcoding. Heuristically use 10% of threads for BAM generation and
    // rest for barcoding. Empirically this shows good perf.
    int barcoder_threads, writer_threads;
    std::tie(barcoder_threads, writer_threads) =
            utils::aligner_writer_thread_allocation(threads, 0.1f);
    spdlog::debug("> barcoding threads {}, writer threads {}", barcoder_threads, writer_threads);

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

    HtsReader reader(reads[0]);
    auto header = sam_hdr_dup(reader.header);
    add_pg_hdr(header);

    PipelineDescriptor pipeline_desc;
    auto hts_writer = pipeline_desc.add_node<BarcodeDemuxer>({}, output_dir, writer_threads, 0);
    std::vector<std::string> kit_names;
    if (parser.present("--kit_name")) {
        kit_names.push_back(parser.get<std::string>("--kit_name"));
    };
    auto barcoder = pipeline_desc.add_node<BarcoderNode>({hts_writer}, barcoder_threads, kit_names);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& hts_writer_ref = dynamic_cast<BarcodeDemuxer&>(pipeline->get_node_ref(hts_writer));
    hts_writer_ref.set_header(header);

    // Set up stats counting
    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(0, false);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables);
    // End stats counting setup.

    spdlog::info("> starting barcoding");
    reader.read(*pipeline, max_reads);

    stats_sampler->terminate();
    auto final_stats = pipeline->terminate();
    tracker.update_progress_bar(final_stats);
    tracker.summarize();

    spdlog::info("> finished barcoding");

    return 0;
}

}  // namespace dorado
