#pragma once

#include <cstdint>
#include <string>

struct sam_hdr_t;
struct bam1_t;
struct htsFile;

namespace dorado::utils {

class HtsFile {
    htsFile* m_file{nullptr};
    sam_hdr_t* m_header{nullptr};

public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsFile(const std::string& filename, OutputMode mode, size_t threads);
    ~HtsFile();

    int set_and_write_header(const sam_hdr_t* const header);
    int write(bam1_t* const record);
};

}  // namespace dorado::utils
