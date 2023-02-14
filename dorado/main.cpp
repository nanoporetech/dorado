#include "Version.h"
#include "cli/cli.h"
#include "spdlog/cfg/env.h"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using entry_ptr = std::function<int(int, char**)>;

namespace {

void usage(const std::vector<std::string> commands) {
    std::cerr << "Usage: dorado [options] subcommand\n\n"
              << "Positional arguments:" << std::endl;

    for (const auto command : commands) {
        std::cerr << command << std::endl;
    }

    std::cerr << "\nOptional arguments:\n"
              << "-h --help               shows help message and exits\n"
              << "-v --version            prints version information and exits\n"
              << "-vv                     prints verbose version information and exits"
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Load logging settings from environment/command-line.
    spdlog::cfg::load_env_levels();

    const std::map<std::string, entry_ptr> subcommands = {
            {"basecaller", &dorado::basecaller},
            {"duplex", &dorado::duplex},
            {"download", &dorado::download},
    };

    std::vector<std::string> arguments(argv + 1, argv + argc);
    std::vector<std::string> keys;

    for (const auto& [key, _] : subcommands) {
        keys.push_back(key);
    }

    if (arguments.size() == 0) {
        usage(keys);
        return 0;
    }

    const auto subcommand = arguments[0];

    if (subcommand == "-v" || subcommand == "--version") {
        std::cerr << DORADO_VERSION << std::endl;
    } else if (subcommand == "-vv") {
#ifdef __APPLE__
        std::cerr << "dorado:   " << DORADO_VERSION << std::endl;
#else
        std::cerr << "dorado:   " << DORADO_VERSION << "+cu" << CUDA_VERSION << std::endl;
#endif
        std::cerr << "libtorch: " << TORCH_BUILD_VERSION << std::endl;
    } else if (subcommand == "-h" || subcommand == "--help") {
        usage(keys);
        return 0;
    } else if (subcommands.find(subcommand) != subcommands.end()) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        usage(keys);
        return 1;
    }

    return 0;
}
