#pragma once

#include "hts_utils/sequence_file_format.h"

#include <filesystem>
#include <memory>
#include <string_view>

namespace dorado::hts_io {

struct FastxRecord {
    std::string_view name;
    std::string_view comment;
    std::string_view seq;
    std::string_view qual;
};

class FastxSequentialReader {
    hts_io::SequenceFormatType fmt_{hts_io::SequenceFormatType::UNKNOWN};

    struct Data;
    std::unique_ptr<Data> data_;

public:
    FastxSequentialReader(const std::filesystem::path& fastx_path);

    ~FastxSequentialReader();

    bool get_next(FastxRecord& record);

    hts_io::SequenceFormatType get_format() const;
};

}  // namespace dorado::hts_io
