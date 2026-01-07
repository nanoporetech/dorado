#include "hts_utils/sequence_file_format.h"

#include "utils/string_utils.h"

#include <algorithm>
#include <string>

namespace dorado::hts_io {

SequenceFormatType parse_sequence_format(const std::filesystem::path& in_path) {
    SequenceFormatType fmt = SequenceFormatType::UNKNOWN;

    // Convert the string to lowercase.
    std::string path_str = in_path.string();
    std::transform(std::begin(path_str), std::end(path_str), std::begin(path_str),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (utils::ends_with(path_str, ".bam")) {
        fmt = SequenceFormatType::BAM;

    } else if (utils::ends_with(path_str, ".sam")) {
        fmt = SequenceFormatType::SAM;

    } else if (utils::ends_with(path_str, ".cram")) {
        fmt = SequenceFormatType::CRAM;

    } else if (utils::ends_with(path_str, ".fasta") || utils::ends_with(path_str, ".fa") ||
               utils::ends_with(path_str, ".fna") || utils::ends_with(path_str, ".fasta.gz") ||
               utils::ends_with(path_str, ".fa.gz") || utils::ends_with(path_str, ".fna.gz")) {
        fmt = SequenceFormatType::FASTA;

    } else if (utils::ends_with(path_str, ".fastq") || utils::ends_with(path_str, ".fq") ||
               utils::ends_with(path_str, ".fastq.gz") || utils::ends_with(path_str, ".fq.gz")) {
        fmt = SequenceFormatType::FASTQ;
    }

    return fmt;
}

}  // namespace dorado::hts_io