#pragma once

#include "types.h"

#include <functional>
#include <string>

namespace dorado::utils {

class HtsFile {
public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    using ProgressCallback = std::function<void(size_t percentage)>;

    HtsFile(const std::string& filename, OutputMode mode, size_t threads);
    ~HtsFile();
    HtsFile(const HtsFile&) = delete;
    HtsFile& operator=(const HtsFile&) = delete;

    int set_and_write_header(const sam_hdr_t* header);
    int write(const bam1_t* record);

    bool finalise_is_noop() const { return m_finalise_is_noop; }
    void finalise(const ProgressCallback& progress_callback,
                  int writer_threads,
                  bool sort_if_mapped);

private:
    HtsFilePtr m_file;
    SamHdrPtr m_header;
    size_t m_num_records{0};
    bool m_finalised{false};
    bool m_finalise_is_noop;
    const OutputMode m_mode;
};

}  // namespace dorado::utils
