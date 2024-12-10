#include "hts_file.h"

#include "utils/PostCondition.h"
#include "utils/bam_utils.h"

#include <htslib/bgzf.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace {

constexpr size_t MINIMUM_BUFFER_SIZE{100000};  // The smallest allowed buffer size is 100 KB.
constexpr size_t DEFAULT_BUFFER_SIZE{
        20000000};  // Arbitrary 20 MB. Can be overridden by application code.
constexpr size_t MAX_FILES_FOR_MERGE{512};  // Maximum number of files to merge at once.

bool compare_headers(const dorado::SamHdrPtr& header1, const dorado::SamHdrPtr& header2) {
    return (strcmp(sam_hdr_str(header1.get()), sam_hdr_str(header2.get())) == 0);
}

// BAM tags to add to the read header for fastx output
constexpr std::array fastq_aux_tags{"RG", "st", "DS", "qs"};

}  // namespace

namespace dorado::utils {

struct HtsFile::ProgressUpdater {
    const ProgressCallback* m_progress_callback{nullptr};
    size_t m_from{0}, m_to{0}, m_max{0}, m_last_progress{0};
    ProgressUpdater() = default;
    ProgressUpdater(const ProgressCallback& progress_callback, size_t from, size_t to, size_t max)
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

HtsFile::HtsFile(const std::string& filename, OutputMode mode, int threads, bool sort_bam)
        : m_filename(filename),
          m_threads(int(threads)),
          m_finalise_is_noop(true),
          m_sort_bam(sort_bam),
          m_mode(mode) {
    switch (m_mode) {
    case OutputMode::FASTQ:
        m_file.reset(hts_open(m_filename.c_str(), "wf"));
        for (const auto& tag : fastq_aux_tags) {
            hts_set_opt(m_file.get(), FASTQ_OPT_AUX, tag);
        }
        break;
    case OutputMode::FASTA:
        m_file.reset(hts_open(filename.c_str(), "wF"));
        for (const auto& tag : fastq_aux_tags) {
            hts_set_opt(m_file.get(), FASTQ_OPT_AUX, tag);
        }
        break;
    case OutputMode::BAM:
        if (m_filename != "-" && m_sort_bam) {
            set_buffer_size(DEFAULT_BUFFER_SIZE);
            // We're doing sorted BAM output. We need to indicate this for the
            // finalise method.
            m_finalise_is_noop = false;
        } else {
            m_sort_bam = false;
            m_file.reset(hts_open(m_filename.c_str(), "wb"));
        }
        break;
    case OutputMode::SAM:
        m_file.reset(hts_open(m_filename.c_str(), "w"));
        break;
    case OutputMode::UBAM:
        m_file.reset(hts_open(m_filename.c_str(), "wb0"));
        break;
    default:
        throw std::runtime_error("Unknown output mode selected: " +
                                 std::to_string(static_cast<int>(m_mode)));
    }

    if (m_threads > 0) {
        initialise_threads();
    }
}

void HtsFile::initialise_threads() {
    if (!m_finalise_is_noop) {
        return;
    }
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + m_filename);
    }

    if (m_file->format.compression == bgzf) {
        auto res = bgzf_mt(m_file->fp.bgzf, m_threads, 128);
        if (res < 0) {
            throw std::runtime_error("Could not enable multi threading for BAM generation.");
        }
    }
}

void HtsFile::set_num_threads(std::size_t threads) {
    if (m_threads > 0) {
        throw std::runtime_error("HtsFile num threads cannot be changed if already initialised");
    }

    if (threads < 1) {
        throw std::runtime_error("HtsFile num threads must be greater than 0");
    }

    m_threads = static_cast<int>(threads);
    initialise_threads();
}

HtsFile::~HtsFile() {
    if (!m_finalised) {
        spdlog::error("finalise() not called on a HtsFile.");
    }
}

uint64_t HtsFile::calculate_sorting_key(const bam1_t* record) {
    return (static_cast<uint64_t>(record->core.tid) << 32) | record->core.pos;
}

void HtsFile::set_buffer_size(size_t buff_size) {
    if (buff_size < MINIMUM_BUFFER_SIZE) {
        throw std::runtime_error("The buffer size for sorted BAM output must be at least " +
                                 std::to_string(MINIMUM_BUFFER_SIZE) + " (" +
                                 std::to_string(MINIMUM_BUFFER_SIZE / 1000) + " KB).");
    }
    m_bam_buffer.resize(buff_size);
}

void HtsFile::flush_temp_file(const bam1_t* last_record) {
    if (m_current_buffer_offset == 0 && !last_record) {
        // This handles the case that the last read passed in before calling finalise() has already triggered
        // a flush, or that finalise() was called without ever passing any reads.
        return;
    }
    if (last_record) {
        // We add last_record to our buffer map with offset -1, so that we know where it should be sorted into
        // the output.
        auto sorting_key = calculate_sorting_key(last_record);
        m_buffer_map.insert({sorting_key, -1});
    }

    // Open the file for writing, and write the header. Note that all temp files will have the same header.
    auto file_index = m_temp_files.size();
    auto tempfilename = m_filename + "." + std::to_string(file_index) + ".tmp";
    m_temp_files.push_back(tempfilename);
    m_file.reset(hts_open(tempfilename.c_str(), "wb"));
    if (m_file->format.compression == bgzf) {
        auto res = bgzf_mt(m_file->fp.bgzf, m_threads, 128);
        if (res < 0) {
            throw std::runtime_error("Could not enable multi threading for BAM generation.");
        }
    }
    if (m_mode != OutputMode::FASTQ && m_mode != OutputMode::FASTA) {
        if (sam_hdr_write(m_file.get(), m_header.get()) != 0) {
            throw std::runtime_error("Could not write header to temp file.");
        }
    }

    for (const auto& item : m_buffer_map) {
        // This will give us the offsets into the buffer in sorted order.
        int64_t offset = item.second;
        const bam1_t* record{nullptr};
        if (offset == -1) {
            record = last_record;
        } else {
            if (size_t(offset) + sizeof(bam1_t) > m_bam_buffer.size()) {
                throw std::out_of_range("Index out of bounds in BAM record buffer.");
            }
            record = std::launder(reinterpret_cast<bam1_t*>(m_bam_buffer.data() + offset));
            if (size_t(offset) + sizeof(bam1_t) + size_t(record->l_data) > m_bam_buffer.size()) {
                throw std::out_of_range("Index out of bounds in BAM record buffer.");
            }
        }
        auto res = write_to_file(record);
        if (res < 0) {
            throw std::runtime_error("Error writing to BAM temporary file, error code " +
                                     std::to_string(res));
        }
    }
    m_file.reset();
    m_current_buffer_offset = 0;
    m_buffer_map.clear();
}

// If we are doing sorted BAM output, then when we are done we will have sorted temporary files
// that need to be merged into a single sorted BAM file. If there's only one temporary file, we
// can just rename it. Otherwise we create a new file, merge the temporary files into it, and
// delete the temporary files. In case an error occurs, the temporary files are left on disk, so
// users can recover their data.
void HtsFile::finalise(const ProgressCallback& progress_callback) {
    assert(progress_callback);
    progress_callback(0);
    auto on_return = utils::PostCondition([&] { progress_callback(100); });

    if (std::exchange(m_finalised, true)) {
        spdlog::error("finalise() called twice on a HtsFile. Ignoring second call.");
        return;
    }

    if (m_finalise_is_noop) {
        // No cleanup is required. Just close the open objects and we're done.
        m_header.reset();
        m_file.reset();
        return;
    }

    // If any reads are cached for writing, write out the final temporary file.
    flush_temp_file(nullptr);

    bool file_is_mapped = (sam_hdr_nref(m_header.get()) > 0);
    m_header.reset();

    if (m_temp_files.empty()) {
        // No temporary files have been written. Nothing to do.
        return;
    }

    size_t num_temp_files = m_temp_files.size();
    if (num_temp_files == 1) {
        // We only have 1 temporary file, so just rename it.
        std::filesystem::rename(m_temp_files.back(), m_filename);
        m_temp_files.clear();
        if (file_is_mapped) {
            // We still need to index the sorted BAM file.
            // We can't update the progress while this is ongoing, so it's just going to
            // say 50% complete until it finishes.
            constexpr size_t percent_start_indexing = 50;
            progress_callback(percent_start_indexing);
            if (sam_index_build3(m_filename.c_str(), nullptr, 0, m_threads) < 0) {
                spdlog::error("Failed to build index for file {}", m_filename);
            }
        }
    } else {
        // Otherwise merge the temp files.
        bool merge_complete = merge_temp_files_iteratively(progress_callback);
        if (!merge_complete) {
            spdlog::error("Merging of temporary files failed.");
            return;
        }
    }
}

int HtsFile::set_header(const sam_hdr_t* const header) {
    if (header) {
        m_header.reset(sam_hdr_dup(header));
        if (m_sort_bam) {
            sam_hdr_change_HD(m_header.get(), "SO", "coordinate");
        }
        if (m_file) {
            return sam_hdr_write(m_file.get(), m_header.get());
        }
    }
    return 0;
}

int HtsFile::write(bam1_t* record) {
    remove_fastq_header_tag(record);
    ++m_num_records;
    if (m_file) {
        return write_to_file(record);
    }
    cache_record(record);
    return 0;
}

int HtsFile::write_to_file(const bam1_t* record) {
    // FIXME -- HtsFile is constructed in a state where attempting to write
    // will segfault, since set_header has to have been called
    // in order to set m_header.
    if (m_mode != OutputMode::FASTQ && m_mode != OutputMode::FASTA) {
        assert(m_header);
    }
    return sam_write1(m_file.get(), m_header.get(), record);
}

void HtsFile::cache_record(const bam1_t* record) {
    size_t bytes_required = sizeof(bam1_t) + size_t(record->l_data);
    if (m_current_buffer_offset + bytes_required > m_bam_buffer.size()) {
        // This record won't fit in the buffer, so flush the current buffer, plus this record, to the file.
        flush_temp_file(record);
        return;
    }
    auto sorting_key = calculate_sorting_key(record);
    m_buffer_map.insert({sorting_key, m_current_buffer_offset});

    // Copy the contents of the bam1_t struct into the memory buffer.
    auto record_buff = m_bam_buffer.data() + m_current_buffer_offset;
    memcpy(record_buff, record, sizeof(bam1_t));
    m_current_buffer_offset += sizeof(bam1_t);

    // The data pointed to by the bam1_t::data field is then copied immediately after the struct contents.
    memcpy(m_bam_buffer.data() + m_current_buffer_offset, record->data, record->l_data);

    // We have to tell our buffered object where its copy of the data is.
    bam1_t* buffer_entry = std::launder(reinterpret_cast<bam1_t*>(record_buff));
    buffer_entry->data =
            std::launder(reinterpret_cast<uint8_t*>(m_bam_buffer.data() + m_current_buffer_offset));

    // When we write the cached records, we will use a pointer cast to treat the cached record as a bam1_t
    // object, so we need to round up our buffer offset so that the next entry will be properly aligned.
    m_current_buffer_offset += size_t(record->l_data);
    auto alignment = alignof(bam1_t);
    m_current_buffer_offset = ((m_current_buffer_offset + alignment - 1) / alignment) * alignment;
}

bool HtsFile::merge_temp_files_iteratively(const ProgressCallback& progress_callback) const {
    // For large numbers of files, we need to merge iteratively.
    FileMergeBatcher batcher(m_temp_files, m_filename, MAX_FILES_FOR_MERGE);
    auto num_batches = batcher.num_batches();
    auto progress_multiplier = batcher.get_recursion_level();
    constexpr size_t percent_start_merging = 5;
    progress_callback(percent_start_merging);
    ProgressUpdater update_progress(progress_callback, percent_start_merging, 100,
                                    m_num_records * progress_multiplier);
    for (size_t iter = 0; iter < num_batches; ++iter) {
        if (!merge_temp_files(update_progress, batcher.get_batch(iter),
                              batcher.get_merge_filename(iter))) {
            return false;
        }
    }
    return true;
}

bool HtsFile::merge_temp_files(ProgressUpdater& update_progress,
                               const std::vector<std::string>& temp_files,
                               const std::string& merged_filename) const {
    // This code assumes the headers for the files are all the same. This will be
    // true if the temp-files were created by this class, but it means that this
    // function is not suitable for generic merging of BAM files.
    const size_t num_temp_files = temp_files.size();
    std::vector<HtsFilePtr> in_files(num_temp_files);
    std::vector<BamPtr> top_records(num_temp_files);
    std::vector<uint64_t> top_record_scores(num_temp_files);
    SamHdrPtr header{};
    for (size_t i = 0; i < num_temp_files; ++i) {
        in_files[i].reset(hts_open(temp_files[i].c_str(), "rb"));
        if (bgzf_mt(in_files[i]->fp.bgzf, m_threads, 128) < 0) {
            spdlog::error("Could not enable multi threading for BAM reading.");
            return false;
        }
        SamHdrPtr current_header(sam_hdr_read(in_files[i].get()));
        if (i == 0) {
            header = std::move(current_header);
        } else {
            // Sanity check. Make sure headers match.
            if (!compare_headers(header, current_header)) {
                spdlog::error("Header for temporary file {} does not match other headers.",
                              temp_files[i]);
                return false;
            }
            current_header.reset();
        }
        top_records[i].reset(bam_init1());
        auto res = sam_read1(in_files[i].get(), header.get(), top_records[i].get());
        if (res < 0) {
            spdlog::error("Could not read first record from file {}, error code {}", temp_files[i],
                          res);
            return false;
        }
        top_record_scores[i] = calculate_sorting_key(top_records[i].get());
    }

    // Open the output file, and write the header.
    HtsFilePtr out_file(hts_open(merged_filename.c_str(), "wb"));
    if (bgzf_mt(out_file->fp.bgzf, m_threads, 128) < 0) {
        spdlog::error("Could not enable multi threading for BAM generation.");
        return false;
    }

    SamHdrPtr out_header(sam_hdr_dup(header.get()));
    sam_hdr_change_HD(out_header.get(), "SO", "coordinate");
    if (sam_hdr_write(out_file.get(), out_header.get()) < 0) {
        spdlog::error("Failed to write header for sorted bam file {}", out_file->fn);
        return false;
    }

    // If this is the final iteration, initialise for indexing.
    bool final_iteration = (std::filesystem::path(merged_filename).extension().string() != ".tmp");
    std::string idx_fname = merged_filename + ".bai";
    if (final_iteration) {
        auto res = sam_idx_init(out_file.get(), out_header.get(), 0, idx_fname.c_str());
        if (res < 0) {
            spdlog::error("Could not initialize output file for indexing, error code {}", res);
            return false;
        }
    }

    size_t processed_records = 0;
    size_t files_done = 0;
    while (files_done < num_temp_files) {
        // Find the next file to write a record from.
        uint64_t best_score = std::numeric_limits<uint64_t>::max();
        int best_index = -1;
        for (size_t i = 0; i < num_temp_files; ++i) {
            if (top_records[i]) {
                auto score = top_record_scores[i];
                if (best_index == -1 || score < best_score) {
                    best_score = score;
                    best_index = int(i);
                }
            }
        }
        if (best_index == -1) {
            spdlog::error("Logic error in merging algorithm.");
            return false;
        }

        // Write the record.
        auto res = sam_write1(out_file.get(), out_header.get(), top_records[best_index].get());
        if (res < 0) {
            spdlog::error("Failed to write to sorted file {}, error code {}", out_file->fn, res);
            return false;
        }
        ++processed_records;
        update_progress(processed_records);

        // Load the next record for the file.
        top_records[best_index].reset(bam_init1());
        res = sam_read1(in_files[best_index].get(), header.get(), top_records[best_index].get());
        if (res >= 0) {
            top_record_scores[best_index] = calculate_sorting_key(top_records[best_index].get());
        } else if (res == -1) {
            // EOF reached. Close the file and mark that this file is done.
            top_records[best_index].reset();
            in_files[best_index].reset();
            ++files_done;
        } else if (res < -1) {
            spdlog::error("Error reading record from file {}, error code {}",
                          in_files[best_index]->fn, res);
            return false;
        }
    }

    if (final_iteration) {
        // Write the index file.
        auto res = sam_idx_save(out_file.get());
        if (res < 0) {
            spdlog::error("Could not write index file, error code {}", res);
            return false;
        }
    }

    out_file.reset();

    // If we got this far, merging was successful, so remove the temporary files.
    // If we returned early due to a merging failure, the temporary files will remain.
    for (const auto& temp_file : temp_files) {
        std::filesystem::remove(temp_file);
    }
    return true;
}

FileMergeBatcher::FileMergeBatcher(const std::vector<std::string>& files,
                                   const std::string& final_file,
                                   size_t batch_size)
        : m_current_batch(0),
          m_batch_size(batch_size),
          m_recursion_level(0),
          m_file_path(std::filesystem::path(final_file).parent_path()) {
    if (m_batch_size < 3) {
        throw std::runtime_error("FileMergeBatcher requires a batchsize of at least 3.");
    }
    m_merge_jobs = recursive_batching(files);
    m_merge_jobs.back().merged_file = final_file;
}

size_t FileMergeBatcher::num_batches() const { return m_merge_jobs.size(); }

size_t FileMergeBatcher::get_recursion_level() const { return m_recursion_level; }

const std::vector<std::string>& FileMergeBatcher::get_batch(size_t n) const {
    if (n >= m_merge_jobs.size()) {
        throw std::range_error("Merge job index out of bounds.");
    }
    return m_merge_jobs[n].files;
}

const std::string& FileMergeBatcher::get_merge_filename(size_t n) const {
    if (n >= m_merge_jobs.size()) {
        throw std::range_error("Merge job index out of bounds.");
    }
    return m_merge_jobs[n].merged_file;
}

std::string FileMergeBatcher::make_merged_filename() {
    auto filename_root = (m_file_path / "batch_").string();
    return filename_root + std::to_string(m_current_batch) + ".tmp";
}

std::vector<FileMergeBatcher::MergeJob> FileMergeBatcher::recursive_batching(
        const std::vector<std::string>& files) {
    std::vector<MergeJob> jobs;
    if (files.size() <= m_batch_size) {
        auto final_file = make_merged_filename();
        jobs.push_back({files, final_file});
        ++m_recursion_level;
        ++m_current_batch;
        return jobs;
    }

    size_t count = files.size();
    size_t num_batches = count / m_batch_size;
    if (count % m_batch_size) {
        ++num_batches;
    }
    size_t batch_size = count / num_batches;
    if (count % num_batches) {
        ++batch_size;
    }
    for (size_t k = 0, batch = 0; batch < num_batches; ++batch) {
        auto this_batch_size = batch_size;
        auto batch_out_file = make_merged_filename();
        if (k + batch_size > count) {
            this_batch_size = count - k;
        }
        jobs.push_back({{}, batch_out_file});
        for (size_t i = 0; i < this_batch_size; ++i) {
            jobs.back().files.push_back(files[k + i]);
        }
        k += this_batch_size;
        ++m_current_batch;
    }
    ++m_recursion_level;

    // Edge case: The last batch could have 1 file in it, which is no good.
    if (jobs.back().files.size() == 1) {
        // Shift the last file from the second-to-last batch to the beginning of the last batch.
        std::vector<std::string> new_job_files = {jobs[num_batches - 2].files.back(),
                                                  jobs.back().files.front()};
        jobs[num_batches - 2].files.pop_back();
        jobs.back().files.swap(new_job_files);
        // Note that for a batch-size of 3 or more, each batch will now have 2 or more files in it.
    }

    // And now the recursion.
    std::vector<std::string> merged_files(num_batches);
    for (size_t k = 0; k < num_batches; ++k) {
        merged_files[k] = jobs[k].merged_file;
    }
    auto more_jobs = recursive_batching(merged_files);
    for (size_t k = 0; k < more_jobs.size(); ++k) {
        jobs.push_back({std::move(more_jobs[k].files), std::move(more_jobs[k].merged_file)});
    }
    return jobs;
}

}  // namespace dorado::utils
