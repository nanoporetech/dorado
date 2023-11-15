#include "BarcodeClassifier.h"

#include "utils/alignment_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <edlib.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado {

namespace {

// Create edlib configuration for detecting barcode region
// using the flanks.
EdlibAlignConfig init_edlib_config_for_flanks() {
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_PATH;
    // The Ns are the barcode mask. The M is for the wobble base in the 16S barcode flanks.
    static const EdlibEqualityPair additionalEqualities[7] = {
            {'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}, {'N', 'U'}, {'M', 'A'}, {'M', 'C'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 7;
    return placement_config;
}

// Create edlib configuration for aligning each barcode against
// the detected region.
EdlibAlignConfig init_edlib_config_for_mask() {
    EdlibAlignConfig mask_config = edlibDefaultAlignConfig();
    mask_config.mode = EDLIB_MODE_NW;
    mask_config.task =
            (spdlog::get_level() == spdlog::level::debug) ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;
    return mask_config;
}

// Extract the position of the barcode mask in the read based
// on the local alignment result from edlib.
int extract_mask_location(EdlibAlignResult aln, std::string_view query) {
    int query_cursor = 0;
    int target_cursor = 0;
    for (int i = 0; i < aln.alignmentLength; i++) {
        if (aln.alignment[i] == EDLIB_EDOP_MATCH) {
            query_cursor++;
            target_cursor++;
            if (query[query_cursor] == 'N') {
                break;
            }
        } else if (aln.alignment[i] == EDLIB_EDOP_MISMATCH) {
            query_cursor++;
            target_cursor++;
        } else if (aln.alignment[i] == EDLIB_EDOP_DELETE) {
            target_cursor++;
        } else if (aln.alignment[i] == EDLIB_EDOP_INSERT) {
            query_cursor++;
        }
    }
    return aln.startLocations[0] + target_cursor;
}

// Helper function to locally align the flanks with barcode mask
// against a subsequence of the read (either front or read window)
// and return the alignment, score & barcode position.
std::tuple<EdlibAlignResult, float, int> extract_flank_fit(std::string_view strand,
                                                           std::string_view read,
                                                           int barcode_len,
                                                           const EdlibAlignConfig& placement_config,
                                                           const char* debug_prefix) {
    EdlibAlignResult result = edlibAlign(strand.data(), int(strand.length()), read.data(),
                                         int(read.length()), placement_config);
    float score = 1.f - static_cast<float>(result.editDistance) / (strand.length() - barcode_len);
    spdlog::debug("{} {} score {}", debug_prefix, result.editDistance, score);
    spdlog::debug("\n{}", utils::alignment_to_str(strand.data(), read.data(), result));
    int bc_loc = extract_mask_location(result, strand);
    return {result, score, bc_loc};
}

// Helper function to globally align a barcode to a region
// within the read.
float extract_mask_score(std::string_view barcode,
                         std::string_view read,
                         const EdlibAlignConfig& config,
                         const char* debug_prefix) {
    auto result = edlibAlign(barcode.data(), int(barcode.length()), read.data(), int(read.length()),
                             config);
    float score = 1.f - static_cast<float>(result.editDistance) / barcode.length();
    spdlog::debug("{} {} score {}", debug_prefix, result.editDistance, score);
    spdlog::debug("\n{}", utils::alignment_to_str(barcode.data(), read.data(), result));
    edlibFreeAlignResult(result);
    return score;
}

bool barcode_is_permitted(const BarcodingInfo::FilterSet& allowed_barcodes,
                          const std::string& barcode_name) {
    if (!allowed_barcodes.has_value()) {
        return true;
    }

    auto normalized_barcode_name = barcode_kits::normalize_barcode_name(barcode_name);
    return allowed_barcodes->count(normalized_barcode_name) != 0;
}

}  // namespace

namespace demux {

const int TRIM_LENGTH = 150;
const BarcodeScoreResult UNCLASSIFIED{};

struct BarcodeClassifier::BarcodeCandidates {
    std::vector<std::string> barcodes;
    std::vector<std::string> barcodes_rev;
    std::string top_context;
    std::string top_context_rev;
    std::string bottom_context;
    std::string bottom_context_rev;
    int top_context_front_flank_len;
    int top_context_rear_flank_len;
    int bottom_context_front_flank_len;
    int bottom_context_rear_flank_len;
    std::vector<std::string> barcode_names;
    std::string kit;
};

BarcodeClassifier::BarcodeClassifier(const std::vector<std::string>& kit_names)
        : m_barcode_candidates(generate_candidates(kit_names)) {}

BarcodeClassifier::~BarcodeClassifier() = default;

BarcodeScoreResult BarcodeClassifier::barcode(
        const std::string& seq,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    auto best_barcode =
            find_best_barcode(seq, m_barcode_candidates, barcode_both_ends, allowed_barcodes);
    return best_barcode;
}

// Generate all possible barcode candidates. If kit name is passed
// limit the candidates generated to only the specified kits. This is done
// to frontload some of the computation, such as calculating flanks
// and their reverse complements, barcode sequences and their reverse complements,
// etc.
// Returns a vector all barcode candidates to test the
// input read sequence against.
std::vector<BarcodeClassifier::BarcodeCandidates> BarcodeClassifier::generate_candidates(
        const std::vector<std::string>& kit_names) {
    const auto& kit_info_map = barcode_kits::get_kit_infos();
    const auto& barcodes = barcode_kits::get_barcodes();

    std::vector<std::string> final_kit_names;
    if (kit_names.empty()) {
        for (auto& [kit_name, _] : kit_info_map) {
            final_kit_names.push_back(kit_name);
        }
    } else {
        final_kit_names = kit_names;
    }
    spdlog::debug("> Kits to evaluate: {}", final_kit_names.size());

    std::vector<BarcodeCandidates> candidates;
    for (auto& kit_name : final_kit_names) {
        auto kit_iter = kit_info_map.find(kit_name);
        if (kit_iter == kit_info_map.end()) {
            throw std::runtime_error(kit_name +
                                     " is not a valid barcode kit name. Please run the help "
                                     "command to find out available barcode kits.");
        }
        const auto& kit_info = kit_iter->second;
        BarcodeCandidates as;
        as.kit = kit_name;
        const auto& ref_bc = barcodes.at(kit_info.barcodes[0]);

        std::string bc_mask(ref_bc.length(), 'N');
        as.top_context = kit_info.top_front_flank + bc_mask + kit_info.top_rear_flank;
        as.top_context_rev = utils::reverse_complement(kit_info.top_rear_flank) + bc_mask +
                             utils::reverse_complement(kit_info.top_front_flank);
        as.bottom_context = kit_info.bottom_front_flank + bc_mask + kit_info.bottom_rear_flank;
        as.bottom_context_rev = utils::reverse_complement(kit_info.bottom_rear_flank) + bc_mask +
                                utils::reverse_complement(kit_info.bottom_front_flank);

        for (const auto& bc_name : kit_info.barcodes) {
            const auto& barcode = barcodes.at(bc_name);
            auto barcode_rev = utils::reverse_complement(barcode);

            as.barcodes.push_back(barcode);
            as.barcodes_rev.push_back(std::move(barcode_rev));

            as.barcode_names.push_back(bc_name);
        }
        candidates.push_back(std::move(as));
    }
    return candidates;
}

// Calculate barcode score for the following barcoding scenario:
// Variant 1 (v1)
// 5' >-=====----------------=====-> 3'
//      BCXX_1             RC(BCXX_2)
//
// Variant 2 (v2)
// 3' <-=====----------------=====-< 5'
//    RC(BCXX_1)             BCXX_2
//
// In this scenario, the barcode (and its flanks) ligate to both ends
// of the read. The flank sequence is also different for top and bottom strands.
// So we need to check both ends of the read. Since the barcodes always ligate to
// 5' end of the read, the 3' end of the other strand has the reverse complement
// of that barcode sequence. This leads to 2 variants of the barcode arrangements.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score_different_double_ends(
        std::string_view read_seq,
        const BarcodeCandidates& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand_v1 = as.top_context;
    std::string_view bottom_strand_v1 = as.bottom_context_rev;
    std::string_view top_strand_v2 = as.bottom_context;
    std::string_view bottom_strand_v2 = as.top_context_rev;
    int barcode_len = int(as.barcodes[0].length());

    // Fetch barcode mask locations for variant 1
    auto [top_result_v1, top_flank_score_v1, top_bc_loc_v1] = extract_flank_fit(
            top_strand_v1, read_top, barcode_len, placement_config, "top score v1");
    std::string_view top_mask_v1 = read_top.substr(top_bc_loc_v1, barcode_len);

    auto [bottom_result_v1, bottom_flank_score_v1, bottom_bc_loc_v1] = extract_flank_fit(
            bottom_strand_v1, read_bottom, barcode_len, placement_config, "bottom score v1");
    std::string_view bottom_mask_v1 = read_bottom.substr(bottom_bc_loc_v1, barcode_len);

    // Fetch barcode mask locations for variant 2
    auto [top_result_v2, top_flank_score_v2, top_bc_loc_v2] = extract_flank_fit(
            top_strand_v2, read_top, barcode_len, placement_config, "top score v2");
    std::string_view top_mask_v2 = read_top.substr(top_bc_loc_v2, barcode_len);

    auto [bottom_result_v2, bottom_flank_score_v2, bottom_bc_loc_v2] = extract_flank_fit(
            bottom_strand_v2, read_bottom, barcode_len, placement_config, "bottom score v2");
    std::string_view bottom_mask_v2 = read_bottom.substr(bottom_bc_loc_v2, barcode_len);

    // Find the best variant of the two.
    int total_v1_score = top_result_v1.editDistance + bottom_result_v1.editDistance;
    int total_v2_score = top_result_v2.editDistance + bottom_result_v2.editDistance;

    std::string_view top_mask, bottom_mask;
    if (total_v1_score < total_v2_score) {
        top_mask = top_mask_v1;
        bottom_mask = bottom_mask_v1;
        spdlog::debug("best variant v1");
    } else {
        top_mask = top_mask_v2;
        bottom_mask = bottom_mask_v2;
        spdlog::debug("best variant v2");
    }

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < as.barcodes.size(); i++) {
        auto& barcode = as.barcodes[i];
        auto& barcode_rev = as.barcodes_rev[i];
        auto& barcode_name = as.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }

        spdlog::debug("Checking barcode {}", barcode_name);

        // Calculate barcode scores for v1.
        auto top_mask_result_score_v1 =
                extract_mask_score(barcode, top_mask_v1, mask_config, "top window v1");

        auto bottom_mask_result_score_v1 =
                extract_mask_score(barcode_rev, bottom_mask_v1, mask_config, "bottom window v1");

        BarcodeScoreResult v1;
        v1.top_score = top_mask_result_score_v1;
        v1.bottom_score = bottom_mask_result_score_v1;
        v1.score = std::max(v1.top_score, v1.bottom_score);
        v1.use_top = v1.top_score > v1.bottom_score;
        v1.top_flank_score = top_flank_score_v1;
        v1.bottom_flank_score = bottom_flank_score_v1;
        v1.flank_score = v1.use_top ? top_flank_score_v1 : bottom_flank_score_v1;
        v1.top_barcode_pos = {top_result_v1.startLocations[0], top_result_v1.endLocations[0]};
        v1.bottom_barcode_pos = {bottom_start + bottom_result_v1.startLocations[0],
                                 bottom_start + bottom_result_v1.endLocations[0]};

        // Calculate barcode scores for v2.
        auto top_mask_result_score_v2 =
                extract_mask_score(barcode, top_mask_v2, mask_config, "top window v2");

        auto bottom_mask_result_score_v2 =
                extract_mask_score(barcode_rev, bottom_mask_v2, mask_config, "bottom window v2");

        BarcodeScoreResult v2;
        v2.top_score = top_mask_result_score_v2;
        v2.bottom_score = bottom_mask_result_score_v2;
        v2.score = std::max(v2.top_score, v2.bottom_score);
        v2.use_top = v2.top_score > v2.bottom_score;
        v2.top_flank_score = top_flank_score_v2;
        v2.bottom_flank_score = bottom_flank_score_v2;
        v2.flank_score = v2.use_top ? top_flank_score_v2 : bottom_flank_score_v2;
        v2.top_barcode_pos = {top_result_v2.startLocations[0], top_result_v2.endLocations[0]};
        v2.bottom_barcode_pos = {bottom_start + bottom_result_v2.startLocations[0],
                                 bottom_start + bottom_result_v2.endLocations[0]};

        // The best score is the higher score between the 2 variants.
        const bool var1_is_best = v1.score > v2.score;
        BarcodeScoreResult res = var1_is_best ? v1 : v2;
        res.variant = var1_is_best ? "var1" : "var2";
        res.barcode_name = barcode_name;
        res.kit = as.kit;

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result_v1);
    edlibFreeAlignResult(bottom_result_v1);
    edlibFreeAlignResult(top_result_v2);
    edlibFreeAlignResult(bottom_result_v2);
    return results;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====--------------=====-> 3'
//      BCXXX            RC(BCXXX)
//
// 3' <-=====--------------=====-< 5'
//    RC(BCXXX)           (BCXXX)
//
// In this scenario, the barcode (and its flanks) potentially ligate to both ends
// of the read. But the barcode sequence is the same for both top and bottom strands.
// So we need to check bottom ends of the read. However since barcode sequence is the
// same for top and bottom strands, we simply need to look for the barcode and its
// reverse complement sequence in the top/bottom windows.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score_double_ends(
        std::string_view read_seq,
        const BarcodeCandidates& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand;
    std::string_view bottom_strand;
    top_strand = as.top_context;
    bottom_strand = as.top_context_rev;
    int barcode_len = int(as.barcodes[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_strand, read_top, barcode_len, placement_config, "top score");
    std::string_view top_mask = read_top.substr(top_bc_loc, barcode_len);

    auto [bottom_result, bottom_flank_score, bottom_bc_loc] = extract_flank_fit(
            bottom_strand, read_bottom, barcode_len, placement_config, "bottom score");
    std::string_view bottom_mask = read_bottom.substr(bottom_bc_loc, barcode_len);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < as.barcodes.size(); i++) {
        auto& barcode = as.barcodes[i];
        auto& barcode_rev = as.barcodes_rev[i];
        auto& barcode_name = as.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }
        spdlog::debug("Checking barcode {}", barcode_name);

        auto top_mask_score = extract_mask_score(barcode, top_mask, mask_config, "top window");

        auto bottom_mask_score =
                extract_mask_score(barcode_rev, bottom_mask, mask_config, "bottom window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = as.kit;
        res.top_score = top_mask_score;
        res.bottom_score = bottom_mask_score;
        res.score = std::max(res.top_score, res.bottom_score);
        res.use_top = res.top_score > res.bottom_score;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = bottom_flank_score;
        res.flank_score = res.use_top ? res.top_flank_score : res.bottom_flank_score;
        res.top_barcode_pos = {top_result.startLocations[0], top_result.endLocations[0]};
        res.bottom_barcode_pos = {bottom_start + bottom_result.startLocations[0],
                                  bottom_start + bottom_result.endLocations[0]};

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    edlibFreeAlignResult(bottom_result);
    return results;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====---------------> 3'
//      BCXXX
//
// In this scenario, the barcode (and its flanks) only ligate to the 5' end
// of the read. So we only look for barcode sequence in the top "window" (first
// 150bp) of the read.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score(
        std::string_view read_seq,
        const BarcodeCandidates& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand;
    top_strand = as.top_context;
    int barcode_len = int(as.barcodes[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_strand, read_top, barcode_len, placement_config, "top score");
    std::string_view top_mask = read_top.substr(top_bc_loc, barcode_len);
    spdlog::debug("BC location {}", top_bc_loc);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < as.barcodes.size(); i++) {
        auto& barcode = as.barcodes[i];
        auto& barcode_name = as.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }

        spdlog::debug("Checking barcode {}", barcode_name);

        auto top_mask_score = extract_mask_score(barcode, top_mask, mask_config, "top window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = as.kit;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = -1.f;
        res.flank_score = std::max(res.top_flank_score, res.bottom_flank_score);
        res.top_score = top_mask_score;
        res.bottom_score = -1.f;
        res.score = res.top_score;
        res.use_top = true;
        res.top_barcode_pos = {top_result.startLocations[0], top_result.endLocations[0]};

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    return results;
}

// Score every barcode against the input read and returns the best match,
// or an unclassified match, based on certain heuristics.
BarcodeScoreResult BarcodeClassifier::find_best_barcode(
        const std::string& read_seq,
        const std::vector<BarcodeCandidates>& candidates,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    if (read_seq.length() < TRIM_LENGTH) {
        return UNCLASSIFIED;
    }
    const std::string_view fwd = read_seq;

    // First find best barcode kit.
    const BarcodeCandidates* as;
    if (candidates.size() == 1) {
        as = &candidates[0];
    } else {
        // TODO: Implement finding best kit match.
        throw std::runtime_error("Unimplemented: multiple barcoding kits");
    }

    // Then find the best barcode hit within that kit.
    const auto& kit_info_map = barcode_kits::get_kit_infos();
    std::vector<BarcodeScoreResult> scores;
    auto& kit = kit_info_map.at(as->kit);
    if (kit.double_ends) {
        if (kit.ends_different) {
            auto out = calculate_barcode_score_different_double_ends(fwd, *as, allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        } else {
            auto out = calculate_barcode_score_double_ends(fwd, *as, allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        }
    } else {
        auto out = calculate_barcode_score(fwd, *as, allowed_barcodes);
        scores.insert(scores.end(), out.begin(), out.end());
    }

    if (scores.empty()) {
        return UNCLASSIFIED;
    }

    if (kit.double_ends) {
        // For a double ended barcode, ensure that the best barcode according
        // to the top window and the best barcode according to the bottom window
        // are the same. If they suggest different barcodes confidently, then
        // consider the read unclassified.
        auto best_top_score = std::max_element(
                scores.begin(), scores.end(),
                [](const auto& l, const auto& r) { return l.top_score < r.top_score; });
        auto best_bottom_score = std::max_element(
                scores.begin(), scores.end(),
                [](const auto& l, const auto& r) { return l.bottom_score < r.bottom_score; });
        spdlog::debug("Check double ends: top bc {}, bottom bc {}", best_top_score->barcode_name,
                      best_bottom_score->barcode_name);
        if ((best_top_score->score > 0.7) && (best_bottom_score->score > 0.7) &&
            (best_top_score->barcode_name != best_bottom_score->barcode_name)) {
            return UNCLASSIFIED;
        }
    }

    // Sort the scores windows by their barcode score.
    std::sort(scores.begin(), scores.end(),
              [](const auto& l, const auto& r) { return l.score > r.score; });

    std::stringstream d;
    for (auto& s : scores) {
        d << s.score << " " << s.barcode_name << ", ";
    }
    spdlog::debug("Scores: {}", d.str());
    auto best_score = scores.begin();
    auto are_scores_acceptable = [](const auto& score) {
        return (score.flank_score >= 0.7 && score.score >= 0.6) ||
               (score.score >= 0.7 && score.flank_score >= 0.6) ||
               (score.top_score >= 0.6 && score.bottom_score >= 0.6);
    };

    BarcodeScoreResult out = UNCLASSIFIED;
    if (scores.size() == 1) {
        if (are_scores_acceptable(*best_score)) {
            out = *best_score;
        }
    } else {
        auto second_best_score = std::next(best_score);
        if (best_score->score - second_best_score->score >= 0.1f) {
            const float kMargin = 0.25f;
            if (are_scores_acceptable(*best_score) ||
                (best_score->score - second_best_score->score >= kMargin)) {
                out = *best_score;
            }
        }
    }

    if (barcode_both_ends && kit.double_ends) {
        // For more stringent classification, ensure that both ends of a read
        // have a high score for the same barcode. If not then consider it
        // unclassified.
        if (out.top_score < 0.6 || out.bottom_score < 0.6) {
            return UNCLASSIFIED;
        }
    }

    // If nothing is found, report as unclassified.
    return out;
}

}  // namespace demux

}  // namespace dorado
