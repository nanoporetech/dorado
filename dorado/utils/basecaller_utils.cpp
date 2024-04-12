#include "basecaller_utils.h"

#include <fstream>
#include <optional>

namespace dorado::utils {
std::optional<std::unordered_set<std::string>> load_read_list(const std::string& read_list) {
    std::unordered_set<std::string> read_ids;

    if (read_list == "") {
        return {};
    }

    std::ifstream dataFile;
    dataFile.open(read_list);

    if (!dataFile.is_open()) {
        throw std::runtime_error("Read list does not exist.");
    }
    std::string cell;

    while (std::getline(dataFile, cell)) {
        read_ids.insert(cell);
    }
    return read_ids;
}
}  // namespace dorado::utils
