#include "summary/summary.h"

#include "cli/cli.h"
#include "dorado_version.h"
#include "utils/log_utils.h"
#include "utils/tty_utils.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include <cctype>
#include <csignal>
#include <filesystem>

namespace dorado {

volatile sig_atomic_t interrupt = 0;

int summary(int argc, char *argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("reads")
            .help("SAM/BAM file produced by dorado basecaller.")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string{});
    parser.add_argument("-s", "--separator").default_value(std::string("\t"));
    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .action([&](const auto &) { ++verbosity; })
            .append();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.get<std::string>("reads"));
    auto separator(parser.get<std::string>("separator"));

    if (!reads.empty()) {
        if (!std::filesystem::exists(reads)) {
            spdlog::error("Unable to open file '{}', no such file.", reads);
            return EXIT_FAILURE;
        }

        if (std::filesystem::is_directory(reads)) {
            spdlog::error("Failed to open file '{}', found a directory instead.", reads);
            return EXIT_FAILURE;
        }
    } else if (utils::is_fd_tty(stdin)) {
        // Only allow `reads` to be empty if we're accepting input from a pipe
        std::cout << parser << '\n';
        return EXIT_FAILURE;
    } else {
        reads = "-";
    }

    SummaryData summary;
    summary.set_separator(separator[0]);
    summary.process_file(reads, std::cout);

    return EXIT_SUCCESS;
}

}  // namespace dorado
