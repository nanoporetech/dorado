#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace dorado {

namespace utils {
class SampleSheet;
}

namespace demux {

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
    std::pair<int, int> top_barcode_pos = {-1, -1};
    std::pair<int, int> bottom_barcode_pos = {-1, -1};
};

const ScoreResults UNCLASSIFIED{};

class BarcodeClassifier {
    struct AdapterSequence;

public:
    BarcodeClassifier(const std::vector<std::string>& kit_names);
    ~BarcodeClassifier();

    ScoreResults barcode(const std::string& seq,
                         bool barcode_both_ends,
                         const utils::SampleSheet* const sample_sheet) const;

private:
    const std::vector<AdapterSequence> m_adapter_sequences;

    std::vector<AdapterSequence> generate_adapter_sequence(
            const std::vector<std::string>& kit_names);
    std::vector<ScoreResults> calculate_adapter_score_different_double_ends(
            std::string_view read_seq,
            const AdapterSequence& as,
            const utils::SampleSheet* const sample_sheet) const;
    std::vector<ScoreResults> calculate_adapter_score_double_ends(
            std::string_view read_seq,
            const AdapterSequence& as,
            const utils::SampleSheet* const sample_sheet) const;
    std::vector<ScoreResults> calculate_adapter_score(
            std::string_view read_seq,
            const AdapterSequence& as,
            const utils::SampleSheet* const sample_sheet) const;
    ScoreResults find_best_adapter(const std::string& read_seq,
                                   const std::vector<AdapterSequence>& adapter,
                                   bool barcode_both_ends,
                                   const utils::SampleSheet* const sample_sheet) const;
};

}  // namespace demux

}  // namespace dorado
