#include "duplex_utils.h"

#include <fstream>

namespace dorado::utils {
std::map<std::string, std::string> load_pairs_file(std::string pairs_file) {
    std::ifstream dataFile;
    dataFile.open(pairs_file);

    std::string cell;
    int line = 0;

    std::getline(dataFile, cell);
    std::map<std::string, std::string> template_complement_map;
    while (!dataFile.eof()) {
        char delim = ' ';
        auto delim_pos = cell.find(delim);

        std::string t = cell.substr(0, delim_pos);
        std::string c = cell.substr(delim_pos + 1, delim_pos * 2 - 1);
        template_complement_map[t] = c;

        line++;
        std::getline(dataFile, cell);
    }
    return template_complement_map;
}
}