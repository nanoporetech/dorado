#include "Version.h"
#include "read_pipeline/DuplexCallerNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/duplex_utils.h"

#include <argparse.hpp>

#include <thread>
#include <spdlog/spdlog.h>
#include "utils/log_utils.h"

namespace dorado {

int duplex(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION);
    parser.add_argument("reads_file").help("Basecalled reads.");
    parser.add_argument("pairs_file").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        std::exit(1);
    }

    std::string reads_file = parser.get<std::string>("reads_file");

    std::string pairs_file = parser.get<std::string>("pairs_file");

    spdlog::info("Loading pairs file: " + pairs_file);
    std::map<std::string, std::string> template_complement_map = load_pairs_file(pairs_file);
    spdlog::info("Pairs file loaded");

    spdlog::info("Loading reads: " + reads_file);
    std::map<std::string, std::shared_ptr<Read>> reads = utils::read_bam(reads_file);
    spdlog::info("Reads loaded");

    std::vector<std::string> args(argv, argv + argc);
    bool emit_fastq = parser.get<bool>("--emit-fastq");

    WriterNode writer_node(std::move(args), emit_fastq, 1);
    DuplexCallerNode duplex_caller_node(writer_node, template_complement_map, reads);

    return 0;
}
}