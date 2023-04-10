#pragma once
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <set>
#include <string>

namespace dorado::utils {

using sq_t = std::vector<std::pair<char*, uint32_t>>;
using read_map = std::map<std::string, std::shared_ptr<Read>>;

class Aligner : public MessageSink {
public:
    Aligner(MessageSink& read_sink, const std::string& filename, int threads);
    ~Aligner();
    sq_t sq();
    std::vector<bam1_t*> align(bam1_t* record, mm_tbuf_t* buf);
    std::pair<int, mm_reg1_t*> align(const std::vector<char> seq);

private:
    MessageSink& m_sink;
    size_t m_threads{1};
    std::atomic<size_t> m_active{0};
    std::vector<mm_tbuf_t*> m_tbufs;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    void worker_thread();
    void add_tags(bam1_t*, const mm_reg1_t*, const std::vector<char>&);

    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_idx_t* m_index{nullptr};
    mm_idx_reader_t* m_index_reader{nullptr};
};

class BamReader {
public:
    BamReader(const std::string& filename);
    BamReader(MessageSink& read_sink, const std::string& filename);
    ~BamReader();
    bool read();
    void read(MessageSink& read_sink, int max_reads = -1);
    char* m_format{nullptr};
    bool m_is_aligned{false};
    bam1_t* m_record{nullptr};
    sam_hdr_t* m_header{nullptr};

private:
    htsFile* m_file{nullptr};
};

class BamWriter : public MessageSink {
public:
    BamWriter(const std::string& filename, size_t threads = 1);
    ~BamWriter();
    int write_header(const sam_hdr_t* header, const sq_t seqs);
    int write(bam1_t* record);
    void join();

    size_t m_total{0};
    size_t m_primary{0};
    size_t m_unmapped{0};
    size_t m_secondary{0};
    size_t m_supplementary{0};

    sam_hdr_t* m_header{nullptr};

private:
    htsFile* m_file{nullptr};
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write_hdr_pg();
    int write_hdr_sq(char* name, uint32_t length);
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
read_map read_bam(const std::string& filename, const std::set<std::string>& read_ids);

}  // namespace dorado::utils
