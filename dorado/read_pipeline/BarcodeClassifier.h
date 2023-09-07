#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace dorado {

namespace demux {

struct AdapterSequence {
    std::vector<std::string> adapter;
    std::vector<std::string> adapter_rev;
    std::string top_primer;
    std::string top_primer_rev;
    std::string bottom_primer;
    std::string bottom_primer_rev;
    int top_primer_front_flank_len;
    int top_primer_rear_flank_len;
    int bottom_primer_front_flank_len;
    int bottom_primer_rear_flank_len;
    std::vector<std::string> adapter_name;
    std::string kit;
};

struct ScoreResults {
    float score = -1.f;
    float top_score = -1.f;
    float bottom_score = -1.f;
    float flank_score = -1.f;
    float top_flank_score = -1.f;
    float bottom_flank_score = -1.f;
    bool use_top = false;
    std::string adapter_name = "unclassified";
    std::string kit = "unclassified";
    int barcode_start = -1;
};

const ScoreResults UNCLASSIFIED{};

class BarcodeClassifier {
public:
    BarcodeClassifier(const std::vector<std::string>& kit_names, bool barcode_both_ends);
    ~BarcodeClassifier() = default;

    ScoreResults barcode(const std::string& seq);

private:
    bool m_barcode_both_ends;
    std::vector<AdapterSequence> m_adapter_sequences;

    std::vector<AdapterSequence> generate_adapter_sequence(
            const std::vector<std::string>& kit_names);
    std::vector<ScoreResults> calculate_adapter_score_different_double_ends(
            std::string_view read_seq,
            const AdapterSequence& as);
    std::vector<ScoreResults> calculate_adapter_score_double_ends(std::string_view read_seq,
                                                                  const AdapterSequence& as);
    std::vector<ScoreResults> calculate_adapter_score(std::string_view read_seq,
                                                      const AdapterSequence& as);
    ScoreResults find_best_adapter(const std::string& read_seq,
                                   std::vector<AdapterSequence>& adapter);
};

}  // namespace demux

}  // namespace dorado
