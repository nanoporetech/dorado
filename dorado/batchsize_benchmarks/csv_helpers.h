#pragma once

#include "SpeedEntry.h"

#include <istream>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

namespace dorado::batchsize_benchmarks {

namespace detail {

struct ParsedLine {
    std::string_view gpu_name;
    std::string_view model_name;
    SpeedEntry entry;
};

std::optional<ParsedLine> csv_try_parse_line(std::string_view line);

}  // namespace detail

/// Write out the entries for a given GPU and model to a stream.
void csv_write_entries(std::ostream &out,
                       std::string_view gpu_name,
                       std::string_view model_name,
                       std::span<const SpeedEntry> entries);

/// Read in the entries contained in a stream.
/// @a Callback receives (gpu_name, model_name, entry) for all entries in the stream.
template <typename Callback>
inline void csv_read_lines(std::istream &in, const Callback &callback) {
    std::string line;
    while (std::getline(in, line)) {
        if (auto parsed = detail::csv_try_parse_line(line); parsed.has_value()) {
            callback(parsed->gpu_name, parsed->model_name, parsed->entry);
        }
    }
}

}  // namespace dorado::batchsize_benchmarks
