#include "fai_utils.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dorado::utils {

std::filesystem::path get_fai_path(const std::filesystem::path& in_fastx_fn) {
    char* idx_name = fai_path(in_fastx_fn.string().c_str());
    std::filesystem::path ret;
    if (idx_name) {
        ret = std::filesystem::path(idx_name);
    }
    hts_free(idx_name);
    return ret;
}

bool check_fai_exists(const std::filesystem::path& in_fastx_fn) {
    const std::filesystem::path idx_name = get_fai_path(in_fastx_fn);
    if (std::empty(idx_name)) {
        return false;
    }
    return std::filesystem::exists(idx_name);
}

bool create_fai_index(const std::filesystem::path& in_fastx_fn) {
    if (std::empty(in_fastx_fn)) {
        spdlog::warn("No path specified, cannnot load/create a FAI index!");
        return false;
    }

    if (std::filesystem::is_empty(in_fastx_fn)) {
        spdlog::warn("Input sequence file '{}' is empty. Not generating the index.",
                     in_fastx_fn.string());
        return true;
    }

    const std::filesystem::path idx_name = get_fai_path(in_fastx_fn);
    spdlog::debug("Looking for idx {}", idx_name.string());

    if (std::empty(idx_name)) {
        spdlog::warn("Empty index path generated for input file: '{}'.", in_fastx_fn.string());
        return false;
    }

    if (!std::filesystem::exists(idx_name)) {
        if (fai_build(in_fastx_fn.string().c_str())) {
            spdlog::warn("Failed to build index for file {}", in_fastx_fn.string());
            return false;
        }
        spdlog::debug("Created the FAI index for file: {}", in_fastx_fn.string());
    }

    return true;
}

std::vector<std::pair<std::string, int64_t>> load_seq_lengths(
        const std::filesystem::path& in_fastx_fn) {
    const std::filesystem::path fai_path = get_fai_path(in_fastx_fn);

    std::vector<std::pair<std::string, int64_t>> ret;
    std::string line;
    std::ifstream ifs(fai_path);
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            continue;
        }
        std::string name;
        int64_t length = 0;
        std::istringstream iss(line);
        iss >> name >> length;
        ret.emplace_back(std::move(name), length);
    }
    return ret;
}

}  // namespace dorado::utils
