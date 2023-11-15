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
    struct AdapterSequence;

public:
    BarcodeClassifier(const std::vector<std::string>& kit_names);
    ~BarcodeClassifier();

    BarcodeScoreResult barcode(const std::string& seq,
                               bool barcode_both_ends,
                               const BarcodingInfo::FilterSet& allowed_barcodes) const;

private:
    const std::vector<AdapterSequence> m_adapter_sequences;

    std::vector<AdapterSequence> generate_adapter_sequence(
            const std::vector<std::string>& kit_names);
    std::vector<BarcodeScoreResult> calculate_adapter_score_different_double_ends(
            std::string_view read_seq,
            const AdapterSequence& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_adapter_score_double_ends(
            std::string_view read_seq,
            const AdapterSequence& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_adapter_score(
            std::string_view read_seq,
            const AdapterSequence& as,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    BarcodeScoreResult find_best_adapter(const std::string& read_seq,
                                         const std::vector<AdapterSequence>& adapter,
                                         bool barcode_both_ends,
                                         const BarcodingInfo::FilterSet& allowed_barcodes) const;
};

}  // namespace demux

}  // namespace dorado
