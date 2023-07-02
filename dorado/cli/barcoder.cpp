#include "read_pipeline/Barcoder.h"

#include "Version.h"
#include "minimap.h"
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

int barcoder(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_description(
            "Alignment using minimap2. The outputs are expected to be equivalent to minimap2.\n"
            "The default parameters use the map-ont preset.\n"
            "NOTE: Not all arguments from minimap2 are currently available. Additionally, "
            "parameter names are not finalized and may change.");
    parser.add_argument("reads").help("any HTS format.").nargs(argparse::nargs_pattern::any);
    parser.add_argument("-t", "--threads")
            .help("number of threads for alignment and BAM writing.")
            .default_value(0)
            .scan<'i', int>();
    parser.add_argument("-n", "--max-reads")
            .help("maxium number of reads to process (for debugging).")
            .default_value(1000000)
            .scan<'i', int>();
    parser.add_argument("-k").default_value(6).scan<'i', int>();
    parser.add_argument("-w").default_value(2).scan<'i', int>();
    parser.add_argument("-m").default_value(20).scan<'i', int>();
    parser.add_argument("-q").default_value(0).scan<'i', int>();
    parser.add_argument("--barcodes").help("barcodes file").default_value("");
    parser.add_argument("--kit_name").help("kit name").default_value("");
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
    auto threads(parser.get<int>("threads"));
    auto max_reads(parser.get<int>("max-reads"));
    auto k(parser.get<int>("k"));
    auto w(parser.get<int>("w"));
    auto m(parser.get<int>("m"));
    auto q(parser.get<int>("q"));
    auto bc_file(parser.get<std::string>("--barcodes"));
    auto kit_name(parser.get<std::string>("--kit_name"));

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

    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(0, false);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });

    HtsWriter writer("-", HtsWriter::OutputMode::BAM, writer_threads, 0);
    std::cerr << "FILE " << bc_file << std::endl;
    Barcoder barcoder(writer, {}, aligner_threads, k, w, m, q, bc_file, kit_name);
    HtsReader reader(reads[0]);

    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
    auto header = sam_hdr_dup(reader.header);
    writer.write_header(header);

    // Setup stats counting.
    std::unique_ptr<dorado::stats::StatsSampler> stats_sampler;
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    using dorado::stats::make_stats_reporter;
    stats_reporters.push_back(make_stats_reporter(writer));
    stats_reporters.push_back(make_stats_reporter(barcoder));

    constexpr auto kStatsPeriod = 100ms;
    stats_sampler = std::make_unique<dorado::stats::StatsSampler>(kStatsPeriod, stats_reporters,
                                                                  stats_callables);
    // End stats counting setup.

    spdlog::info("> starting barcoding");
    reader.read(barcoder, max_reads);
    writer.join();

    stats_sampler->terminate();
    tracker.summarize();

    spdlog::info("> finished barcoding");

    return 0;
}

}  // namespace dorado
