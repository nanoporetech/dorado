#include "Version.h"
#include "minimap.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace dorado {

int aligner(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    spdlog::info("MM_MAX_SEG {}", MM_MAX_SEG);

    return 0;
}

}  // namespace dorado
