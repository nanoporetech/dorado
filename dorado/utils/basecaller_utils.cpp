#include "basecaller_utils.h"

#include <fstream>

namespace dorado::utils {
std::unordered_set<std::string> load_read_list(std::string read_list) {
    std::unordered_set<std::string> read_ids;

    if (read_list == "") {
        return read_ids;
    }

    std::ifstream dataFile;
    dataFile.open(read_list);

    if (!dataFile.is_open()) {
        throw std::runtime_error("Read list does not exist.");
    }
    std::string cell;
    int line = 0;

    std::getline(dataFile, cell);
    while (!dataFile.eof()) {
        read_ids.insert(cell);
        line++;
        std::getline(dataFile, cell);
    }
    return read_ids;
}
}  // namespace dorado::utils