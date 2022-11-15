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
    parser.add_argument("model").help("Model");
    parser.add_argument("reads").help("Reads in Pod5 format or BAM/SAM format for basespace.");
    parser.add_argument("--pairs").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);
    parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        std::exit(1);
    }

    std::string model = parser.get<std::string>("model");
    std::string reads = parser.get<std::string>("reads");
    std::string pairs_file = parser.get<std::string>("--pairs");
    size_t threads = static_cast<size_t>(parser.get<int>("--threads"));
    bool emit_fastq = parser.get<bool>("--emit-fastq");
    std::vector<std::string> args(argv, argv + argc);

    if (model.compare("basespace") != 0) {
        spdlog::error("> Unsupported model {}", model);
        return 1;
    }

    spdlog::info("> Loading pairs file");
    std::map<std::string, std::string> template_complement_map = utils::load_pairs_file(pairs_file);

    spdlog::info("> Loading reads");
    std::map<std::string, std::shared_ptr<Read>> read_map = utils::read_bam(reads);

    torch::set_num_threads(1);
    threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
    WriterNode writer_node(std::move(args), emit_fastq, false, true, 4);
    DuplexCallerNode duplex_caller_node(writer_node, template_complement_map, read_map, threads);

    return 0;
}
}  // namespace dorado
