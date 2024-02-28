#pragma once

#include "types.h"

#include <string>

namespace dorado::utils {

class HtsFile {
    HtsFilePtr m_file;
    SamHdrPtr m_header;

public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsFile(const std::string& filename, OutputMode mode, size_t threads);
    ~HtsFile();

    int set_and_write_header(const sam_hdr_t* header);
    int write(const bam1_t* record);
};

}  // namespace dorado::utils
