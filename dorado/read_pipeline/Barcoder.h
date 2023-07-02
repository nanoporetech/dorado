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

struct KitInfo {
    bool fwd_rev_separate;
    std::string fwd_front_flank;
    std::string fwd_rear_flank;
    std::string rev_front_flank;
    std::string rev_rear_flank;
    std::vector<std::string> barcodes;
};

static const std::unordered_map<std::string, KitInfo> kit_info = {
        {"SQK-RBK004",
         {false,
          "GCTTGGGTGTTTAACC",
          "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA",
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10", "BC11",
           "BC12"}}},
        {"SQK-RBK114.24",
         {false,
          "GCTTGGGTGTTTAACC",
          "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA",
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08",
           "BC09", "BC10", "BC11", "BC12", "BC13", "BC14", "BC15", "BC16",
           "BC17", "BC18", "BC19", "BC20", "BC21", "BC22", "BC23", "BC24"}}},
        {"SQK-RBK110.96",
         {false,
          "GCTTGGGTGTTTAACC",
          "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA",
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10", "BC11",
           "BC12", "BC13", "BC14", "BC15", "BC16", "BC17", "BC18", "BC19", "BC20", "BC21", "BC22",
           "BC23", "BC24", "BC25", "BC26", "BC27", "BC28", "BC29", "BC30", "BC31", "BC32", "BC33",
           "BC34", "BC35", "BC36", "BC37", "BC38", "BC39", "BC40", "BC41", "BC42", "BC43", "BC44",
           "BC45", "BC46", "BC47", "BC48", "BC49", "BC50", "BC51", "BC52", "BC53", "BC54", "BC55",
           "BC56", "BC57", "BC58", "BC59", "BC60", "BC61", "BC62", "BC63", "BC64", "BC65", "BC66",
           "BC67", "BC68", "BC69", "BC70", "BC71", "BC72", "BC73", "BC74", "BC75", "BC76", "BC77",
           "BC78", "BC79", "BC80", "BC81", "BC82", "BC83", "BC84", "BC85", "BC86", "BC87", "BC88",
           "BC89", "BC90", "BC91", "BC92", "BC93", "BC94", "BC95", "BC96"}}},
        {"SQK-RBK114.96",
         {false,
          "GCTTGGGTGTTTAACC",
          "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA",
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10", "BC11",
           "BC12", "BC13", "BC14", "BC15", "BC16", "BC17", "BC18", "BC19", "BC20", "BC21", "BC22",
           "BC23", "BC24", "BC25", "BC26", "BC27", "BC28", "BC29", "BC30", "BC31", "BC32", "BC33",
           "BC34", "BC35", "BC36", "BC37", "BC38", "BC39", "BC40", "BC41", "BC42", "BC43", "BC44",
           "BC45", "BC46", "BC47", "BC48", "BC49", "BC50", "BC51", "BC52", "BC53", "BC54", "BC55",
           "BC56", "BC57", "BC58", "BC59", "BC60", "BC61", "BC62", "BC63", "BC64", "BC65", "BC66",
           "BC67", "BC68", "BC69", "BC70", "BC71", "BC72", "BC73", "BC74", "BC75", "BC76", "BC77",
           "BC78", "BC79", "BC80", "BC81", "BC82", "BC83", "BC84", "BC85", "BC86", "BC87", "BC88",
           "BC89", "BC90", "BC91", "BC92", "BC93", "BC94", "BC95", "BC96"}}},
        {"SQK-RPB004",
         {false,
          "ATCGCCTACCGTGAC",
          "CGTTTTTCGTGCGCCGCTTC",
          "",
          "",
          {"BC01", "BC02", "BC03", "BC04", "BC05", "BC06", "BC07", "BC08", "BC09", "BC10", "BC11",
           "RLB12A"}}},
};

}  // namespace barcoding

using sq_t = std::vector<std::pair<char*, uint32_t>>;

class Barcoder : public MessageSink {
public:
    Barcoder(MessageSink& read_sink,
             const std::vector<std::string>& barcodes,
             int threads,
             int k,
             int w,
             int m,
             int q,
             const std::string& barcode_file,
             const std::string& kit_name);
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
    std::vector<BamPtr> barcode(bam1_t* irecord, mm_tbuf_t* buf);
    std::atomic<int> m_matched{0};
    int m_q;

    void init_mm2_settings(int k, int w, int m);
    std::string mm2_barcode(const std::string& seq, const std::string_view& qname, mm_tbuf_t* buf);
    std::string edlib_barcode(const std::string& seq, const std::string& seq_rev);

    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_idx_t* m_index{nullptr};

    void read_barcodes(const std::string& barcode_file);

    std::unordered_map<std::string, std::string> m_barcodes;
    std::string m_kit_name;
};

}  // namespace dorado
