#include "fai_utils.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

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

void create_fai_index(const std::filesystem::path& in_fastx_fn) {
    if (std::empty(in_fastx_fn)) {
        throw std::runtime_error{"Cannot load/create a FAI index from an empty path!"};
    }

    const std::filesystem::path idx_name = get_fai_path(in_fastx_fn);
    spdlog::debug("Looking for idx {}", idx_name.string());

    if (std::empty(idx_name)) {
        throw std::runtime_error{"Empty index path generated for input file: '" +
                                 in_fastx_fn.string() + "'."};
    }

    if (!std::filesystem::exists(idx_name)) {
        if (fai_build(in_fastx_fn.string().c_str())) {
            spdlog::error("Failed to build index for file {}", in_fastx_fn.string());
            throw std::runtime_error{"Failed to build index for file " + in_fastx_fn.string() +
                                     "."};
        }
        spdlog::debug("Created the FAI index for file: {}", in_fastx_fn.string());
    }
}

}  // namespace dorado::utils
