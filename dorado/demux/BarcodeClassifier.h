#pragma once
#include "barcoding_info.h"
#include "utils/barcode_kits.h"
#include "utils/parse_custom_kit.h"
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
    BarcodeClassifier(const std::vector<std::string>& kit_names,
                      const std::optional<std::string>& custom_kit,
                      const std::optional<std::string>& custom_sequences);
    ~BarcodeClassifier();

    BarcodeScoreResult barcode(const std::string& seq,
                               bool barcode_both_ends,
                               const BarcodingInfo::FilterSet& allowed_barcodes) const;

private:
    const std::unordered_map<std::string, dorado::barcode_kits::KitInfo> m_custom_kit;
    const std::unordered_map<std::string, std::string> m_custom_seqs;
    const barcode_kits::BarcodeKitScoringParams m_scoring_params;
    const std::vector<BarcodeCandidateKit> m_barcode_candidates;

    std::vector<BarcodeCandidateKit> generate_candidates(const std::vector<std::string>& kit_names);
    float find_midstrand_barcode_different_double_ends(std::string_view read_seq,
                                                       const BarcodeCandidateKit& candidate) const;
    float find_midstrand_barcode_double_ends(std::string_view read_seq,
                                             const BarcodeCandidateKit& candidate) const;
    float find_midstrand_barcode_single_end(std::string_view read_seq,
                                            const BarcodeCandidateKit& candidate) const;
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
            const BarcodingInfo::FilterSet& allowed_barcodes) const;
    BarcodeScoreResult find_best_barcode(const std::string& read_seq,
                                         const std::vector<BarcodeCandidateKit>& adapter,
                                         bool barcode_both_ends,
                                         const BarcodingInfo::FilterSet& allowed_barcodes) const;

    const dorado::barcode_kits::KitInfo& get_kit_info(const std::string& kit_name) const;
    const std::string& get_barcode_sequence(const std::string& barcode_name) const;
};

}  // namespace demux

}  // namespace dorado
