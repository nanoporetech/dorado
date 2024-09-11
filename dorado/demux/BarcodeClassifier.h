#pragma once
#include "KitInfoProvider.h"
#include "barcoding_info.h"
#include "utils/barcode_kits.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dorado {

namespace demux {

class BarcodeClassifier {
    struct BarcodeCandidateKit;

public:
    BarcodeClassifier(KitInfoProvider kit_info_provider);

    BarcodeClassifier(const std::vector<std::string>& kit_names,
                      const std::optional<std::string>& custom_kit,
                      const std::optional<std::string>& custom_sequences)
            : BarcodeClassifier(KitInfoProvider(kit_names, custom_kit, custom_sequences)) {};

    ~BarcodeClassifier();

    BarcodeScoreResult barcode(const std::string& seq,
                               bool barcode_both_ends,
                               const BarcodingInfo::FilterSet& allowed_barcodes) const;

private:
    const KitInfoProvider m_kit_info_provider;
    const barcode_kits::BarcodeKitScoringParams m_scoring_params;
    const std::vector<BarcodeCandidateKit> m_barcode_candidates;

    std::vector<BarcodeCandidateKit> generate_candidates();
    float find_midstrand_barcode_different_double_ends(std::string_view read_seq,
                                                       const BarcodeCandidateKit& candidate) const;
    float find_midstrand_barcode_double_ends(std::string_view read_seq,
                                             const BarcodeCandidateKit& candidate) const;
    float find_midstrand_barcode_single_end(std::string_view read_seq,
                                            const BarcodeCandidateKit& candidate,
                                            bool rear_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_barcode_score_different_double_ends(
            std::string_view read_seq,
            const BarcodeCandidateKit& candidate,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_barcode_score_double_ends(
            std::string_view read_seq,
            const BarcodeCandidateKit& candidate,
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    std::vector<BarcodeScoreResult> calculate_barcode_score(
            std::string_view read_seq,
            const BarcodeCandidateKit& candidate,
            const BarcodingInfo::FilterSet& allowed_barcodes,
            bool rear_barcodes) const;
    BarcodeScoreResult find_best_barcode(const std::string& read_seq,
                                         const std::vector<BarcodeCandidateKit>& adapter,
                                         bool barcode_both_ends,
                                         const BarcodingInfo::FilterSet& allowed_barcodes) const;
};

}  // namespace demux

}  // namespace dorado
