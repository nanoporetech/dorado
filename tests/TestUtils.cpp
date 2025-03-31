#include "TestUtils.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace dorado::tests {

std::filesystem::path get_data_dir(const std::string& sub_dir) {
    const std::filesystem::path root_path("./tests/data/");

    // clang-tidy warns about performance-no-automatic-move if |data_path| is const. It should be treated as such though.
    /*const*/ auto data_path = root_path / sub_dir;
    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the root directory of the dorado checkout\n";
        exit(1);
    }
    return data_path;
}

TempDir make_temp_dir(const std::string& prefix) {
#ifdef _WIN32
    std::filesystem::path path;
    while (true) {
        char temp[L_tmpnam];
        const char* name = std::tmpnam(temp);
        auto filename = std::filesystem::path(name);
        filename.replace_filename(prefix + std::string("_") + filename.filename().string());
        if (std::filesystem::create_directories(filename)) {
            path = std::filesystem::canonical(filename);
            break;
        }
    }
#else
    // macOS (rightfully) complains about tmpnam() usage, so make use of mkdtemp() on platforms that support it
    std::string temp = (std::filesystem::temp_directory_path() / (prefix + "_XXXXXXXXXX")).string();
    const char* name = mkdtemp(temp.data());
    auto path = std::filesystem::canonical(name);
#endif
    return TempDir(std::move(path));
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

std::string generate_random_sequence_string(int len) {
    const char bases[4] = {'A', 'C', 'G', 'T'};
    std::string read(len, 'A');
    for (int i = 0; i < len; i++) {
        read[i] = bases[rand() % 4];
    }
    return read;
}

}  // namespace dorado::tests
