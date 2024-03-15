#pragma once

#include "types.h"

#include <string>

namespace dorado::utils {

class HtsFile {
    HtsFilePtr m_file;
    SamHdrPtr m_header;
    size_t m_num_records{0};
    bool m_finalised{false};
    bool m_finalise_is_noop;

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

    bool finalise_is_noop() const { return m_finalise_is_noop; }
    void finalise();
};

}  // namespace dorado::utils
