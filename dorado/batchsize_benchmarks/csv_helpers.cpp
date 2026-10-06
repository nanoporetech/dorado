#include "csv_helpers.h"

#include "utils/SeparatedStream.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"

namespace dorado::batchsize_benchmarks {

namespace detail {

std::optional<ParsedLine> csv_try_parse_line(std::string_view line) {
    // Ignore empty lines.
    if (line.empty()) {
        return std::nullopt;
    }

    // Split up this line.
    utils::SeparatedStream<','> stream(line);
    const auto gpu_name = stream.getline();
    const auto model_name = stream.getline();
    const auto batch_size_str = stream.getline();
    const auto basecall_speed_str = stream.getline();
    const auto memory_used_str = stream.getline();
    if (stream.eof()) {
        // Ignore this line if any of the components are missing.
        DORADO_LOG(warn, "Line doesn't have enough entries: {}", line);
        return std::nullopt;
    }

    // Check that the GPU and model are set.
    if (gpu_name.value().empty() || model_name.value().empty()) {
        DORADO_LOG(warn, "Failed to parse line: {}", line);
        return std::nullopt;
    }

    // Parse numbers.
    const auto batch_size = utils::from_chars<std::uint32_t>(batch_size_str.value());
    const auto basecall_speed = utils::from_chars<double>(basecall_speed_str.value());
    const auto memory_used = utils::from_chars<std::uint64_t>(memory_used_str.value());
    if (!batch_size.has_value() || !basecall_speed.has_value() || !memory_used.has_value()) {
        // Ignore this line if the values failed to parse.
        DORADO_LOG(warn, "Failed to parse line: {}", line);
        return std::nullopt;
    }

    const SpeedEntry entry{
            .batch_size = batch_size.value(),
            .basecall_speed = basecall_speed.value(),
            .memory_used = memory_used.value(),
    };
    return ParsedLine{
            .gpu_name = gpu_name.value(),
            .model_name = model_name.value(),
            .entry = entry,
    };
}

}  // namespace detail

void csv_write_entries(std::ostream &out,
                       std::string_view gpu_name,
                       std::string_view model_name,
                       std::span<const SpeedEntry> entries) {
    for (const auto &entry : entries) {
        out << gpu_name << ',';
        out << model_name << ',';
        out << entry.batch_size << ',';
        out << entry.basecall_speed << ',';
        out << entry.memory_used << '\n';
    }
}

}  // namespace dorado::batchsize_benchmarks
