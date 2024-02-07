#include "TestUtils.h"

#include <cstring>
#include <fstream>
#include <iostream>

namespace dorado::tests {

std::string get_data_dir(const std::string& sub_dir) {
    const std::filesystem::path data_path = std::filesystem::path("./tests/data/") / sub_dir;

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the build working directory inside dorado"
                  << std::endl;
        exit(1);
    }
    return data_path.string();
}

std::string ReadFileIntoString(const std::filesystem::path& path) {
    const auto num_bytes = std::filesystem::file_size(path);
    std::string content;
    content.resize(num_bytes);
    std::ifstream in_file(path.c_str(), std::ios::in | std::ios::binary);
    in_file.read(content.data(), content.size());
    return content;
}

std::vector<uint8_t> ReadFileIntoVector(const std::filesystem::path& path) {
    const std::string str = ReadFileIntoString(path);
    std::vector<uint8_t> vec;
    vec.resize(str.size());
    std::memcpy(vec.data(), str.data(), str.size());
    return vec;
}

}  // namespace dorado::tests
