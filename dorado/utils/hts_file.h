#pragma once

#include "types.h"

#include <algorithm>
#include <functional>
#include <map>
#include <string>

namespace dorado::utils {

class HtsFile {
public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
        FASTA,
    };

    using ProgressCallback = std::function<void(size_t percentage)>;

    HtsFile(const std::string& filename, OutputMode mode, int threads, bool sort_bam);
    ~HtsFile();
    HtsFile(const HtsFile&) = delete;
    HtsFile& operator=(const HtsFile&) = delete;

    // Support for setting threads after construction
    void set_num_threads(std::size_t threads);

    void set_buffer_size(size_t buff_size);
    int set_header(const sam_hdr_t* header);
    int write(bam1_t* record);

    bool finalise_is_noop() const { return m_finalise_is_noop; }
    void finalise(const ProgressCallback& progress_callback);
    static uint64_t calculate_sorting_key(const bam1_t* record);

    OutputMode get_output_mode() const { return m_mode; }

private:
    std::string m_filename;
    HtsFilePtr m_file;
    SamHdrPtr m_header;
    size_t m_num_records{0};
    int m_threads{0};
    bool m_finalised{false};
    bool m_finalise_is_noop;
    bool m_sort_bam;
    const OutputMode m_mode;

    std::vector<std::byte> m_bam_buffer;
    std::multimap<uint64_t, int64_t> m_buffer_map;
    std::vector<std::string> m_temp_files;
    int64_t m_current_buffer_offset{0};

    struct ProgressUpdater;

    void flush_temp_file(const bam1_t* last_record);
    int write_to_file(const bam1_t* record);
    void cache_record(const bam1_t* record);
    bool merge_temp_files(ProgressUpdater& update_progress) const;
    void initialise_threads();
};

}  // namespace dorado::utils
