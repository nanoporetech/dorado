#pragma once
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>

namespace dorado {

using sq_t = std::vector<std::pair<char*, uint32_t>>;

class Aligner : public MessageSink {
public:
    Aligner(MessageSink& read_sink,
            const std::string& filename,
            int k,
            int w,
            uint64_t index_batch_size,
            int threads);
    ~Aligner();
    std::string get_name() const override { return "Aligner"; }
    stats::NamedStats sample_stats() const override;
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

}  // namespace dorado
