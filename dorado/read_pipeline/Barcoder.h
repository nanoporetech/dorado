#pragma once
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>
#include <vector>

namespace dorado {

namespace barcoding {

static const std::unordered_map<std::string, std::string> barcodes = {
        {"RLB01", "AAGAAAGTTGTCGGTGTCTTTGTG"}, {"RLB02", "TCGATTCCGTTTGTAGTCGTCTGT"},
        {"RLB03", "GAGTCTTGTGTCCCAGTTACCAGG"}, {"RLB04", "TTCGGATTCTATCGTGTTTCCCTA"},
        {"RLB05", "CTTGTCCAGGGTTTGTGTAACCTT"}, {"RLB06", "TTCTCGCAAAGGCAGAAAGTAGTC"},
        {"RLB07", "GTGTTACCGTGGGAATGAATCCTT"}, {"RLB08", "TTCAGGGAACAAACCAAGTTACGT"},
        {"RLB09", "AACTAGGCACAGCGAGTCTTGGTT"}, {"RLB10", "AAGCGTTGAAACCTTTGTCCTCTC"},
        {"RLB11", "GTTTCATCTATCGGAGGGAATGGA"}, {"RLB12", "CAGGTAGAAAGAAGCAGAATCGGA"},
        {"RLB13", "GTTGAGTTACAAAGCACCGATCAG"},
};

}  // namespace barcoding

using sq_t = std::vector<std::pair<char*, uint32_t>>;

class Barcoder : public MessageSink {
public:
    Barcoder(MessageSink& read_sink, const std::vector<std::string>& barcodes, int threads);
    ~Barcoder();
    std::string get_name() const override { return "Barcoder"; }
    stats::NamedStats sample_stats() const override;

private:
    MessageSink& m_sink;
    size_t m_threads{1};
    std::atomic<size_t> m_active{0};
    std::vector<mm_tbuf_t*> m_tbufs;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    void worker_thread(size_t tid);
    void add_tags(bam1_t* record, const mm_reg1_t* aln);
    std::vector<BamPtr> align(bam1_t* irecord, mm_tbuf_t* buf);
    std::atomic<int> matched{0};

    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_idx_t* m_index{nullptr};
};

}  // namespace dorado
