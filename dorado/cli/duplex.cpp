#include "Version.h"
#include "read_pipeline/DuplexCallerNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/duplex_utils.h"

#include <argparse.hpp>

#include <iostream>
#include <thread>
/*
#include "3rdparty/edlib/edlib/include/edlib.h"
*/

int duplex(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);
    parser.add_argument("reads_file").help("Basecalled reads.");
    parser.add_argument("pairs_file").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    std::cerr << "Loading BAM" << std::endl;

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::string reads_file = parser.get<std::string>("reads_file");
    std::string pairs_file = parser.get<std::string>("pairs_file");

    // Load the pairs file
    std::map<std::string, std::string> template_complement_map = load_pairs_file(pairs_file);

    // Load basecalls
    std::map<std::string, std::shared_ptr<Read>> reads = read_bam(reads_file);

    std::vector<std::string> args(argv, argv + argc);
    bool emit_fastq = parser.get<bool>("--emit-fastq");

    WriterNode writer_node(std::move(args), emit_fastq, 1);
    DuplexCallerNode duplex_caller_node(writer_node, template_complement_map, reads);

    return 0;
}