#pragma once
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace dorado {

namespace demux {

class BarcodeClassifier {
    struct BarcodeCandidateKit;

public:
    BarcodeClassifier(const std::vector<std::string>& kit_names);
    ~BarcodeClassifier();

    BarcodeScoreResult barcode(const std::string& seq,
                               bool barcode_both_ends,
                               const BarcodingInfo::FilterSet& allowed_barcodes) const;

private:
    const std::vector<BarcodeCandidateKit> m_barcode_candidates;

    std::vector<BarcodeCandidateKit> generate_candidates(const std::vector<std::string>& kit_names);
    std::vector<BarcodeScoreResult> calculate_barcode_score_different_double_ends(
            std::string_view read_seq,
            const BarcodeCandidateKit& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_barcode_score_double_ends(
            std::string_view read_seq,
            const BarcodeCandidateKit& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_barcode_score(
            std::string_view read_seq,
            const BarcodeCandidateKit& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    BarcodeScoreResult find_best_barcode(const std::string& read_seq,
                                         const std::vector<BarcodeCandidateKit>& adapter,
                                         bool barcode_both_ends,
                                         const BarcodingInfo::FilterSet& allowed_barcodes) const;
};

}  // namespace demux

}  // namespace dorado
