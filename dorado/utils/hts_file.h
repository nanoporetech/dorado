#pragma once

#include "types.h"

#include <algorithm>
#include <filesystem>
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
    bool merge_temp_files_iteratively(const ProgressCallback& progress_callback) const;
    bool merge_temp_files(ProgressUpdater& update_progress,
                          const std::vector<std::string>& temp_files,
                          const std::string& merged_filename) const;
    void initialise_threads();
};

class FileMergeBatcher {
public:
    FileMergeBatcher(const std::vector<std::string>& files,
                     const std::string& final_file,
                     size_t batch_size);
    size_t num_batches() const;
    size_t get_recursion_level() const;
    const std::vector<std::string>& get_batch(size_t n) const;
    const std::string& get_merge_filename(size_t n) const;

private:
    struct MergeJob {
        std::vector<std::string> files;
        std::string merged_file;
    };

    int m_current_batch;
    size_t m_batch_size;
    size_t m_recursion_level;
    std::vector<MergeJob> m_merge_jobs;
    std::filesystem::path m_file_path;

    std::string make_merged_filename();
    std::vector<MergeJob> recursive_batching(const std::vector<std::string>& files);
};

}  // namespace dorado::utils
