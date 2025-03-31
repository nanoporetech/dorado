#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>

namespace dorado::utils {

/**
 * \brief Returns a pointer to an output stream.
 *          If the `out_fn` is empty, the returned pointer is to stdout, otherwise
 *          a file is opened and a pointer to the output stream returned.
 */
std::unique_ptr<std::ostream, void (*)(std::ostream*)> get_output_stream(
        const std::filesystem::path& out_fn);

}  // namespace dorado::utils
