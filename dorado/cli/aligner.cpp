#include "Version.h"
#include "minimap.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

namespace dorado {

int aligner(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("index").help("reference in (fastq/fasta/mmi).");
    parser.add_argument("reads").help("any HTS format.").nargs(argparse::nargs_pattern::any);
    parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();
    parser.add_argument("-n", "--max-reads").default_value(1000).scan<'i', int>();
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

    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    spdlog::debug("> threads {}", threads);

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

    utils::BamWriter writer("-");
    utils::Aligner aligner(writer, index, threads);
    utils::BamReader reader(reads[0]);

    spdlog::debug("> input fmt: {} aligned: {}", reader.m_format, reader.m_is_aligned);
    writer.write_header(reader.m_header, aligner.sq());

    spdlog::info("> starting alignment");
    reader.read(aligner, max_reads);
    writer.join();

    spdlog::info("> finished alignment");
    spdlog::info("> total/primary/unmapped {}/{}/{}", writer.m_total, writer.m_primary,
                 writer.m_unmapped);

    return 0;
}

}  // namespace dorado
