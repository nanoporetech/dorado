#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <functional>

#include "Version.h"
#include "cli/cli.h"


using entry_ptr = std::function<int(int, char**)>;


int usage(std::vector<std::string> commands) {

    std::ostringstream ss;
    const char* const delim = ", ";
    std::copy(commands.begin(), commands.end(), std::ostream_iterator<std::string>(ss, delim));

    std::cout << "usage: dorado [-h] [-v] {" << ss.str() << "} ...\n\n"
              << "optional arguments:\n"
              << "  -h, --help            show this help message and exit\n"
              << "  -v, --version         show program's version number and exit\n\n"
              << "subcommands:\n"
              << "  valid commands\n\n"
              << "  {" << ss.str() << "}" << std::endl;
     return 0;
}


int main(int argc, char *argv[]) {

    std::map<std::string, entry_ptr> subcommands = {
        {"basecaller", &basecaller},
    };

    std::vector<std::string> arguments(argv + 1, argv + argc);
    std::vector<std::string> keys;

    for (const auto& [key, _] : subcommands) {
         keys.push_back(key);
    }

    if (arguments.size() == 0) {
        return usage(keys);
    }

    auto subcommand = arguments[0];

    if (subcommand == "-v" || subcommand == "--version") {
        std::cout << DORADO_VERSION << std::endl;
    } else if (subcommands.contains(subcommand)) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        return usage(keys);
    }

}
