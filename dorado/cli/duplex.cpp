#include "Version.h"
#include "read_pipeline/DuplexCallerNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/duplex_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <thread>

namespace dorado {

int duplex(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION);
    parser.add_argument("bam_file").help("Basecalled reads in BAM/SAM format.");
    parser.add_argument("pairs_file").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);
    parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        std::exit(1);
    }

    std::string bam_file = parser.get<std::string>("bam_file");
    std::string pairs_file = parser.get<std::string>("pairs_file");
    size_t threads = static_cast<size_t>(parser.get<int>("--threads"));
    bool emit_fastq = parser.get<bool>("--emit-fastq");
    std::vector<std::string> args(argv, argv + argc);

    spdlog::info("> Loading pairs file");
    std::map<std::string, std::string> template_complement_map = utils::load_pairs_file(pairs_file);

    spdlog::info("> Loading reads");
    std::map<std::string, std::shared_ptr<Read>> reads = utils::read_bam(bam_file);

    torch::set_num_threads(1);
    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;

    WriterNode writer_node(std::move(args), emit_fastq, false, 4);
    DuplexCallerNode duplex_caller_node(writer_node, template_complement_map, reads, threads);

    return 0;
}
}  // namespace dorado
