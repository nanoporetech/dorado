#pragma once

#include "types.h"

#include <string>

namespace dorado::utils {

class HtsFile {
    HtsFilePtr m_file;
    SamHdrPtr m_header;
    bool m_finalised{false};

public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsFile(const std::string& filename, OutputMode mode, size_t threads);
    ~HtsFile();
    HtsFile(const HtsFile&) = delete;
    HtsFile& operator=(const HtsFile&) = delete;

    int set_and_write_header(const sam_hdr_t* header);
    int write(const bam1_t* record);

    void finalise();
};

}  // namespace dorado::utils
