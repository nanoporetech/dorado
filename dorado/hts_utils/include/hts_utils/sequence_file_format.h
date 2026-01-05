#pragma once

#include <filesystem>

namespace dorado::hts_io {

enum class SequenceFormatType {
    BAM,
    FASTA,
    FASTQ,
    SAM,
    CRAM,
    UNKNOWN,
};

SequenceFormatType parse_sequence_format(const std::filesystem::path& in_path);

}  // namespace dorado::hts_io
