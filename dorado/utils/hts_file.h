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
    };

    using ProgressCallback = std::function<void(size_t percentage)>;

    HtsFile(const std::string& filename, OutputMode mode, size_t threads, bool sort_bam);
    ~HtsFile();
    HtsFile(const HtsFile&) = delete;
    HtsFile& operator=(const HtsFile&) = delete;

    void set_buffer_size(size_t buff_size);
    int set_header(const sam_hdr_t* header);
    int write(const bam1_t* record);

    bool finalise_is_noop() const { return m_finalise_is_noop; }
    void finalise(const ProgressCallback& progress_callback);
    static uint64_t calculate_sorting_key(const bam1_t* record);

    struct ProgressUpdater {
        const ProgressCallback* m_progress_callback{nullptr};
        size_t m_from{0}, m_to{0}, m_max{0}, m_last_progress{0};
        ProgressUpdater() = default;
        ProgressUpdater(const ProgressCallback& progress_callback,
                        size_t from,
                        size_t to,
                        size_t max)
                : m_progress_callback(&progress_callback),
                  m_from(from),
                  m_to(to),
                  m_max(max),
                  m_last_progress(0) {}

        void operator()(size_t count) {
            if (!m_progress_callback) {
                return;
            }
            const size_t new_progress = m_from + (m_to - m_from) * std::min(count, m_max) / m_max;
            if (new_progress != m_last_progress) {
                m_last_progress = new_progress;
                m_progress_callback->operator()(new_progress);
            }
        }
    };

    OutputMode get_output_mode() const { return m_mode; }

private:
    std::string m_filename;
    HtsFilePtr m_file;
    SamHdrPtr m_header;
    size_t m_num_records{0};
    int m_threads{0};
    bool m_finalised{false};
    bool m_finalise_is_noop;
    bool m_sort_bam{false};
    const OutputMode m_mode;

    std::vector<uint8_t> m_bam_buffer;
    std::multimap<uint64_t, int64_t> m_buffer_map;
    std::vector<std::string> m_temp_files;
    int64_t m_current_buffer_offset{0};
    size_t m_buffer_size{0};

    void flush_temp_file(const bam1_t* last_record);
    int write_to_file(const bam1_t* record);
    int cache_record(const bam1_t* record);
    bool merge_temp_files(ProgressUpdater& update_progress);
};

}  // namespace dorado::utils
