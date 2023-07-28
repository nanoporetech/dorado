#pragma once
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

class Aligner : public MessageSink {
public:
    // header_sequence_records is populated by the constructor.
    Aligner(const std::string& filename, int k, int w, uint64_t index_batch_size, int threads);
    ~Aligner();
    std::string get_name() const override { return "Aligner"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    using bam_header_sq_t = std::vector<std::pair<char*, uint32_t>>;
    bam_header_sq_t get_sequence_records_for_header() const;

private:
    void start_threads();
    void terminate_impl();
    size_t m_threads{1};
    std::vector<mm_tbuf_t*> m_tbufs;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    void worker_thread(size_t tid);
    void add_tags(bam1_t*, const mm_reg1_t*, const std::string&, const mm_tbuf_t*);
    std::vector<BamPtr> align(bam1_t* record, mm_tbuf_t* buf);

    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_idx_t* m_index{nullptr};
    mm_idx_reader_t* m_index_reader{nullptr};
};

}  // namespace dorado
