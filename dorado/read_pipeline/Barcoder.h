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

struct KitInfo {
    bool fwd_rev_separate;
    std::string top_front_flank;
    std::string top_rear_flank;
    std::string bottom_front_flank;
    std::string bottom_rear_flank;
    std::vector<std::string> barcodes;
};

struct AdapterSequence {
    std::string adapter;
    std::string adapter_rev;
    std::string top_primer;
    std::string top_primer_rev;
    std::string bottom_primer;
    std::string bottom_primer_rev;
    int top_primer_flank_len;
    int bottom_primer_flank_len;
    std::string adapter_name;
    std::string kit;
};

struct ScoreResults {
    int score;
    std::string adapter_name;
    std::string kit;
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

static const std::unordered_map<std::string, std::string> barcodes = {
        {"BC01", "AAGAAAGTTGTCGGTGTCTTTGTG"},   {"BC02", "TCGATTCCGTTTGTAGTCGTCTGT"},
        {"BC03", "GAGTCTTGTGTCCCAGTTACCAGG"},   {"BC04", "TTCGGATTCTATCGTGTTTCCCTA"},
        {"BC05", "CTTGTCCAGGGTTTGTGTAACCTT"},   {"BC06", "TTCTCGCAAAGGCAGAAAGTAGTC"},
        {"BC07", "GTGTTACCGTGGGAATGAATCCTT"},   {"BC08", "TTCAGGGAACAAACCAAGTTACGT"},
        {"BC09", "AACTAGGCACAGCGAGTCTTGGTT"},   {"BC10", "AAGCGTTGAAACCTTTGTCCTCTC"},
        {"BC11", "GTTTCATCTATCGGAGGGAATGGA"},   {"BC12", "CAGGTAGAAAGAAGCAGAATCGGA"},
        {"RLB12A", "GTTGAGTTACAAAGCACCGATCAG"}, {"BC13", "AGAACGACTTCCATACTCGTGTGA"},
        {"BC14", "AACGAGTCTCTTGGGACCCATAGA"},   {"BC15", "AGGTCTACCTCGCTAACACCACTG"},
        {"BC16", "CGTCAACTGACAGTGGTTCGTACT"},   {"BC17", "ACCCTCCAGGAAAGTACCTCTGAT"},
        {"BC18", "CCAAACCCAACAACCTAGATAGGC"},   {"BC19", "GTTCCTCGTGCAGTGTCAAGAGAT"},
        {"BC20", "TTGCGTCCTGTTACGAGAACTCAT"},   {"BC21", "GAGCCTCTCATTGTCCGTTCTCTA"},
        {"BC22", "ACCACTGCCATGTATCAAAGTACG"},   {"BC23", "CTTACTACCCAGTGAACCTCCTCG"},
        {"BC24", "GCATAGTTCTGCATGATGGGTTAG"},   {"BC25", "GTAAGTTGGGTATGCAACGCAATG"},
        {"BC26", "CATACAGCGACTACGCATTCTCAT"},   {"BC27", "CGACGGTTAGATTCACCTCTTACA"},
        {"BC28", "TGAAACCTAAGAAGGCACCGTATC"},   {"BC29", "CTAGACACCTTGGGTTGACAGACC"},
        {"BC30", "TCAGTGAGGATCTACTTCGACCCA"},   {"BC31", "TGCGTACAGCAATCAGTTACATTG"},
        {"BC32", "CCAGTAGAAGTCCGACAACGTCAT"},   {"BC33", "CAGACTTGGTACGGTTGGGTAACT"},
        {"BC34", "GGACGAAGAACTCAAGTCAAAGGC"},   {"BC35", "CTACTTACGAAGCTGAGGGACTGC"},
        {"BC36", "ATGTCCCAGTTAGAGGAGGAAACA"},   {"BC37", "GCTTGCGATTGATGCTTAGTATCA"},
        {"BC38", "ACCACAGGAGGACGATACAGAGAA"},   {"BC39", "CCACAGTGTCAACTAGAGCCTCTC"},
        {"BC40", "TAGTTTGGATGACCAAGGATAGCC"},   {"BC41", "GGAGTTCGTCCAGAGAAGTACACG"},
        {"BC42", "CTACGTGTAAGGCATACCTGCCAG"},   {"BC43", "CTTTCGTTGTTGACTCGACGGTAG"},
        {"BC44", "AGTAGAAAGGGTTCCTTCCCACTC"},   {"BC45", "GATCCAACAGAGATGCCTTCAGTG"},
        {"BC46", "GCTGTGTTCCACTTCATTCTCCTG"},   {"BC47", "GTGCAACTTTCCCACAGGTAGTTC"},
        {"BC48", "CATCTGGAACGTGGTACACCTGTA"},   {"BC49", "ACTGGTGCAGCTTTGAACATCTAG"},
        {"BC50", "ATGGACTTTGGTAACTTCCTGCGT"},   {"BC51", "GTTGAATGAGCCTACTGGGTCCTC"},
        {"BC52", "TGAGAGACAAGATTGTTCGTGGAC"},   {"BC53", "AGATTCAGACCGTCTCATGCAAAG"},
        {"BC54", "CAAGAGCTTTGACTAAGGAGCATG"},   {"BC55", "TGGAAGATGAGACCCTGATCTACG"},
        {"BC56", "TCACTACTCAACAGGTGGCATGAA"},   {"BC57", "GCTAGGTCAATCTCCTTCGGAAGT"},
        {"BC58", "CAGGTTACTCCTCCGTGAGTCTGA"},   {"BC59", "TCAATCAAGAAGGGAAAGCAAGGT"},
        {"BC60", "CATGTTCAACCAAGGCTTCTATGG"},   {"BC61", "AGAGGGTACTATGTGCCTCAGCAC"},
        {"BC62", "CACCCACACTTACTTCAGGACGTA"},   {"BC63", "TTCTGAAGTTCCTGGGTCTTGAAC"},
        {"BC64", "GACAGACACCGTTCATCGACTTTC"},   {"BC65", "TTCTCAGTCTTCCTCCAGACAAGG"},
        {"BC66", "CCGATCCTTGTGGCTTCTAACTTC"},   {"BC67", "GTTTGTCATACTCGTGTGCTCACC"},
        {"BC68", "GAATCTAAGCAAACACGAAGGTGG"},   {"BC69", "TACAGTCCGAGCCTCATGTGATCT"},
        {"BC70", "ACCGAGATCCTACGAATGGAGTGT"},   {"BC71", "CCTGGGAGCATCAGGTAGTAACAG"},
        {"BC72", "TAGCTGACTGTCTTCCATACCGAC"},   {"BC73", "AAGAAACAGGATGACAGAACCCTC"},
        {"BC74", "TACAAGCATCCCAACACTTCCACT"},   {"BC75", "GACCATTGTGATGAACCCTGTTGT"},
        {"BC76", "ATGCTTGTTACATCAACCCTGGAC"},   {"BC77", "CGACCTGTTTCTCAGGGATACAAC"},
        {"BC78", "AACAACCGAACCTTTGAATCAGAA"},   {"BC79", "TCTCGGAGATAGTTCTCACTGCTG"},
        {"BC80", "CGGATGAACATAGGATAGCGATTC"},   {"BC81", "CCTCATCTTGTGAAGTTGTTTCGG"},
        {"BC82", "ACGGTATGTCGAGTTCCAGGACTA"},   {"BC83", "TGGCTTGATCTAGGTAAGGTCGAA"},
        {"BC84", "GTAGTGGACCTAGAACCTGTGCCA"},   {"BC85", "AACGGAGGAGTTAGTTGGATGATC"},
        {"BC86", "AGGTGATCCCAACAAGCGTAAGTA"},   {"BC87", "TACATGCTCCTGTTGTTAGGGAGG"},
        {"BC88", "TCTTCTACTACCGATCCGAAGCAG"},   {"BC89", "ACAGCATCAATGTTTGGCTAGTTG"},
        {"BC90", "GATGTAGAGGGTACGGTTTGAGGC"},   {"BC91", "GGCTCCATAGGAACTCACGCTACT"},
        {"BC92", "TTGTGAGTGGAAAGATACAGGACC"},   {"BC93", "AGTTTCCATCACTTCAGACTTGGG"},
        {"BC94", "GATTGTCCTCAAACTGCCACCTAC"},   {"BC95", "CCTGTCTGGAAGAAGAATGGACTT"},
        {"BC96", "CTGAACGGTCATAGAGTCCACCAT"},
};

using sq_t = std::vector<std::pair<char*, uint32_t>>;

class Barcoder : public MessageSink {
public:
    Barcoder(MessageSink& read_sink,
             const std::vector<std::string>& barcodes,
             int threads,
             const std::string& barcode_file,
             const std::string& kit_name);
    ~Barcoder();
    std::string get_name() const override { return "Barcoder"; }
    stats::NamedStats sample_stats() const override;

private:
    MessageSink& m_sink;
    size_t m_threads{1};
    std::atomic<size_t> m_active{0};
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<int> m_matched{0};
    std::string m_kit_name;

    void worker_thread(size_t tid);
    std::vector<BamPtr> barcode(bam1_t* irecord);
    std::vector<AdapterSequence> generate_adapter_sequence(
            const std::vector<std::string>& kit_names);
    ScoreResults calculate_adapter_score(const std::string_view& read_seq,
                                         const std::string_view& read_seq_rev,
                                         const AdapterSequence& as,
                                         bool with_flanks);
    ScoreResults find_best_adapter(const std::string& read_seq,
                                   std::vector<AdapterSequence>& adapter);
};

}  // namespace dorado
