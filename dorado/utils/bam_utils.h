#pragma once
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/types.h"

#include <indicators/block_progress_bar.hpp>

#include <map>
#include <set>
#include <string>

namespace dorado::utils {

using sq_t = std::vector<std::pair<char*, uint32_t>>;
using read_map = std::unordered_map<std::string, std::shared_ptr<Read>>;

class Aligner : public MessageSink {
public:
    Aligner(MessageSink& read_sink, const std::string& filename, int k, int w, int threads);
    ~Aligner();
    std::vector<BamPtr> align(bam1_t* record, mm_tbuf_t* buf);
    sq_t get_sequence_records_for_header();

private:
    MessageSink& m_sink;
    size_t m_threads{1};
    std::atomic<size_t> m_active{0};
    std::vector<mm_tbuf_t*> m_tbufs;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    void worker_thread(size_t tid);
    void add_tags(bam1_t*, const mm_reg1_t*, const std::string&, const mm_tbuf_t*);

    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_idx_t* m_index{nullptr};
    mm_idx_reader_t* m_index_reader{nullptr};
};

class HtsReader {
public:
    HtsReader(const std::string& filename);
    ~HtsReader();
    bool read();
    void read(MessageSink& read_sink, int max_reads = -1);

    char* format{nullptr};
    bool is_aligned{false};
    BamPtr record{nullptr};
    sam_hdr_t* header{nullptr};

private:
    htsFile* m_file{nullptr};
};

class HtsWriter : public MessageSink {
public:
    enum OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsWriter(const std::string& filename, OutputMode mode, size_t threads, size_t num_reads);
    ~HtsWriter();
    void add_header(const sam_hdr_t* header);
    int write_header();
    int write(bam1_t* record);
    void join();

    static OutputMode get_output_mode(std::string mode);

    size_t total{0};
    size_t primary{0};
    size_t unmapped{0};
    size_t secondary{0};
    size_t supplementary{0};
    sam_hdr_t* header{nullptr};

private:
    htsFile* m_file{nullptr};
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write_hdr_sq(char* name, uint32_t length);

    size_t m_num_reads_expected;
    int m_progress_bar_interval;
    indicators::BlockProgressBar m_progress_bar{
            indicators::option::Stream{std::cerr},     indicators::option::BarWidth{30},
            indicators::option::ShowElapsedTime{true}, indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage{true},
    };
};

/**
 * @brief Reads a SAM/BAM/CRAM file and returns a map of read IDs to Read objects.
 *
 * This function opens a SAM/BAM/CRAM file specified by the input filename parameter,
 * reads the alignments, and creates a map that associates read IDs with their
 * corresponding Read objects. The Read objects contain the read ID, sequence,
 * and quality string.
 *
 * @param filename The input BAM file path as a string.
 * @param read_ids A set of read_ids to filter on.
 * @return A map with read IDs as keys and shared pointers to Read objects as values.
 *
 * @note The caller is responsible for managing the memory of the returned map.
 * @note The input BAM file must be properly formatted and readable.
 */
read_map read_bam(const std::string& filename, const std::unordered_set<std::string>& read_ids);

void add_rg_hdr(sam_hdr_t* hdr, const std::unordered_map<std::string, ReadGroup>& read_groups);

void add_sq_hdr(sam_hdr_t* hdr, const sq_t& seqs);

}  // namespace dorado::utils
