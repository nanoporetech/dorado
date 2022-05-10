#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <functional>

#include "Version.h"
#include "cli/cli.h"


using entry_ptr = std::function<int(int, char**)>;


void usage(const std::vector<std::string> commands) {

    std::cerr << "Usage: dorado [options] subcommand\n\n"
              << "Positional arguments:" << std::endl;

    for (const auto command : commands) {
        std::cerr << command << std::endl;
    }

    std::cerr << "\nOptional arguments:\n"
              << "-h --help               shows help message and exits\n"
              << "-v --version            prints version information and exits"
              << std::endl;

}


int main(int argc, char *argv[]) {

    const std::map<std::string, entry_ptr> subcommands = {
        {"basecaller", &basecaller},
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
    } else if (subcommands.contains(subcommand)) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        usage(keys);
        return 1;
    }

    return 0;

}
