#include "io_utils.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>

namespace dorado::utils {

std::unique_ptr<std::ostream, void (*)(std::ostream*)> get_output_stream(
        const std::filesystem::path& out_fn) {
    if (std::empty(out_fn)) {
        return {&std::cout, [](std::ostream*) {}};
    }
    std::unique_ptr<std::ofstream, void (*)(std::ostream*)> ofs(
            new std::ofstream(out_fn), [](std::ostream* ptr) { delete ptr; });
    if (!ofs->is_open()) {
        throw std::runtime_error("Failed to open file: " + out_fn.string());
    }
    return ofs;
}

}  // namespace dorado::utils
