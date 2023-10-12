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
                                                           int adapter_len,
                                                           const EdlibAlignConfig& placement_config,
                                                           const char* debug_prefix) {
    EdlibAlignResult result = edlibAlign(strand.data(), int(strand.length()), read.data(),
                                         int(read.length()), placement_config);
    float score = 1.f - static_cast<float>(result.editDistance) / (strand.length() - adapter_len);
    spdlog::debug("{} {} score {}", debug_prefix, result.editDistance, score);
    spdlog::debug("\n{}", utils::alignment_to_str(strand.data(), read.data(), result));
    int bc_loc = extract_mask_location(result, strand);
    return {result, score, bc_loc};
}

// Helper function to globally align a barcode to a region
// within the read.
float extract_mask_score(std::string_view adapter,
                         std::string_view read,
                         const EdlibAlignConfig& config,
                         const char* debug_prefix) {
    auto result = edlibAlign(adapter.data(), int(adapter.length()), read.data(), int(read.length()),
                             config);
    float score = 1.f - static_cast<float>(result.editDistance) / adapter.length();
    spdlog::debug("{} {} score {}", debug_prefix, result.editDistance, score);
    spdlog::debug("\n{}", utils::alignment_to_str(adapter.data(), read.data(), result));
    edlibFreeAlignResult(result);
    return score;
}

bool barcode_is_permitted(const BarcodingInfo::FilterSet& allowed_barcodes,
                          const std::string& adapter_name) {
    if (!allowed_barcodes.has_value()) {
        return true;
    }

    auto normalized_barcode_name = barcode_kits::normalize_barcode_name(adapter_name);
    return allowed_barcodes->count(normalized_barcode_name) != 0;
}

}  // namespace

namespace demux {

const int TRIM_LENGTH = 150;
const BarcodeScoreResult UNCLASSIFIED{};

struct BarcodeClassifier::AdapterSequence {
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

BarcodeClassifier::BarcodeClassifier(const std::vector<std::string>& kit_names)
        : m_adapter_sequences(generate_adapter_sequence(kit_names)) {}

BarcodeClassifier::~BarcodeClassifier() = default;

BarcodeScoreResult BarcodeClassifier::barcode(
        const std::string& seq,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    auto best_adapter =
            find_best_adapter(seq, m_adapter_sequences, barcode_both_ends, allowed_barcodes);
    return best_adapter;
}

// Generate all possible barcode adapters. If kit name is passed
// limit the adapters generated to only the specified kits. This is done
// to frontload some of the computation, such as calculating flanks
// and their reverse complements, adapters and their reverse complements,
// etc.
// Returns a vector all barcode adapter sequences to test the
// input read sequence against.
std::vector<BarcodeClassifier::AdapterSequence> BarcodeClassifier::generate_adapter_sequence(
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

    std::vector<AdapterSequence> adapters;
    for (auto& kit_name : final_kit_names) {
        auto kit_iter = kit_info_map.find(kit_name);
        if (kit_iter == kit_info_map.end()) {
            throw std::runtime_error(kit_name +
                                     " is not a valid barcode kit name. Please run the help "
                                     "command to find out available barcode kits.");
        }
        const auto& kit_info = kit_iter->second;
        AdapterSequence as;
        as.kit = kit_name;
        const auto& ref_bc = barcodes.at(kit_info.barcodes[0]);

        std::string bc_mask(ref_bc.length(), 'N');
        as.top_primer = kit_info.top_front_flank + bc_mask + kit_info.top_rear_flank;
        as.top_primer_rev = utils::reverse_complement(kit_info.top_rear_flank) + bc_mask +
                            utils::reverse_complement(kit_info.top_front_flank);
        as.bottom_primer = kit_info.bottom_front_flank + bc_mask + kit_info.bottom_rear_flank;
        as.bottom_primer_rev = utils::reverse_complement(kit_info.bottom_rear_flank) + bc_mask +
                               utils::reverse_complement(kit_info.bottom_front_flank);

        for (const auto& bc_name : kit_info.barcodes) {
            const auto& adapter = barcodes.at(bc_name);
            auto adapter_rev = utils::reverse_complement(adapter);

            as.adapter.push_back(adapter);
            as.adapter_rev.push_back(std::move(adapter_rev));

            as.adapter_name.push_back(bc_name);
        }
        adapters.push_back(std::move(as));
    }
    return adapters;
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
// of the read. The adapter sequence is also different for top and bottom strands.
// So we need to check both ends of the read. Since the adapters always ligate to
// 5' end of the read, the 3' end of the other strand has the reverse complement
// of that adapter sequence. This leads to 2 variants of the barcode arrangements.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_adapter_score_different_double_ends(
        std::string_view read_seq,
        const AdapterSequence& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand_v1 = as.top_primer;
    std::string_view bottom_strand_v1 = as.bottom_primer_rev;
    std::string_view top_strand_v2 = as.bottom_primer;
    std::string_view bottom_strand_v2 = as.top_primer_rev;
    int adapter_len = int(as.adapter[0].length());

    // Fetch barcode mask locations for variant 1
    auto [top_result_v1, top_flank_score_v1, top_bc_loc_v1] = extract_flank_fit(
            top_strand_v1, read_top, adapter_len, placement_config, "top score v1");
    std::string_view top_mask_v1 = read_top.substr(top_bc_loc_v1, adapter_len);

    auto [bottom_result_v1, bottom_flank_score_v1, bottom_bc_loc_v1] = extract_flank_fit(
            bottom_strand_v1, read_bottom, adapter_len, placement_config, "bottom score v1");
    std::string_view bottom_mask_v1 = read_bottom.substr(bottom_bc_loc_v1, adapter_len);

    // Fetch barcode mask locations for variant 2
    auto [top_result_v2, top_flank_score_v2, top_bc_loc_v2] = extract_flank_fit(
            top_strand_v2, read_top, adapter_len, placement_config, "top score v2");
    std::string_view top_mask_v2 = read_top.substr(top_bc_loc_v2, adapter_len);

    auto [bottom_result_v2, bottom_flank_score_v2, bottom_bc_loc_v2] = extract_flank_fit(
            bottom_strand_v2, read_bottom, adapter_len, placement_config, "bottom score v2");
    std::string_view bottom_mask_v2 = read_bottom.substr(bottom_bc_loc_v2, adapter_len);

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
    for (size_t i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_rev = as.adapter_rev[i];
        auto& adapter_name = as.adapter_name[i];

        if (!barcode_is_permitted(allowed_barcodes, adapter_name)) {
            continue;
        }

        spdlog::debug("Checking barcode {}", adapter_name);

        // Calculate barcode scores for v1.
        auto top_mask_result_score_v1 =
                extract_mask_score(adapter, top_mask_v1, mask_config, "top window v1");

        auto bottom_mask_result_score_v1 =
                extract_mask_score(adapter_rev, bottom_mask_v1, mask_config, "bottom window v1");

        BarcodeScoreResult v1;
        v1.top_score = top_mask_result_score_v1;
        v1.bottom_score = bottom_mask_result_score_v1;
        v1.score = std::max(v1.top_score, v1.bottom_score);
        v1.use_top = v1.top_score > v1.bottom_score;
        v1.top_flank_score = top_flank_score_v1;
        v1.bottom_flank_score = bottom_flank_score_v1;
        v1.flank_score = v1.use_top ? top_flank_score_v1 : bottom_flank_score_v1;
        v1.barcode_start = v1.use_top ? top_bc_loc_v1 : bottom_start + bottom_bc_loc_v1;
        v1.top_barcode_pos = {top_result_v1.startLocations[0], top_result_v1.endLocations[0]};
        v1.bottom_barcode_pos = {bottom_start + bottom_result_v1.startLocations[0],
                                 bottom_start + bottom_result_v1.endLocations[0]};

        // Calculate barcode scores for v2.
        auto top_mask_result_score_v2 =
                extract_mask_score(adapter, top_mask_v2, mask_config, "top window v2");

        auto bottom_mask_result_score_v2 =
                extract_mask_score(adapter_rev, bottom_mask_v2, mask_config, "bottom window v2");

        BarcodeScoreResult v2;
        v2.top_score = top_mask_result_score_v2;
        v2.bottom_score = bottom_mask_result_score_v2;
        v2.score = std::max(v2.top_score, v2.bottom_score);
        v2.use_top = v2.top_score > v2.bottom_score;
        v2.top_flank_score = top_flank_score_v2;
        v2.bottom_flank_score = bottom_flank_score_v2;
        v2.flank_score = v2.use_top ? top_flank_score_v2 : bottom_flank_score_v2;
        v2.barcode_start = v2.use_top ? top_bc_loc_v2 : bottom_start + bottom_bc_loc_v2;
        v2.top_barcode_pos = {top_result_v2.startLocations[0], top_result_v2.endLocations[0]};
        v2.bottom_barcode_pos = {bottom_start + bottom_result_v2.startLocations[0],
                                 bottom_start + bottom_result_v2.endLocations[0]};

        // The best score is the higher score between the 2 variants.
        BarcodeScoreResult res = (v1.score > v2.score) ? v1 : v2;
        res.adapter_name = adapter_name;
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
// of the read. But the adapter sequence is the same for both top and bottom strands.
// So we need to check bottom ends of the read. However since adapter sequence is the
// same for top and bottom strands, we simply need to look for the adapter and its
// reverse complement sequence in the top/bottom windows.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_adapter_score_double_ends(
        std::string_view read_seq,
        const AdapterSequence& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand;
    std::string_view bottom_strand;
    top_strand = as.top_primer;
    bottom_strand = as.top_primer_rev;
    int adapter_len = int(as.adapter[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_strand, read_top, adapter_len, placement_config, "top score");
    std::string_view top_mask = read_top.substr(top_bc_loc, adapter_len);

    auto [bottom_result, bottom_flank_score, bottom_bc_loc] = extract_flank_fit(
            bottom_strand, read_bottom, adapter_len, placement_config, "bottom score");
    std::string_view bottom_mask = read_bottom.substr(bottom_bc_loc, adapter_len);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_rev = as.adapter_rev[i];
        auto& adapter_name = as.adapter_name[i];

        if (!barcode_is_permitted(allowed_barcodes, adapter_name)) {
            continue;
        }
        spdlog::debug("Checking barcode {}", adapter_name);

        auto top_mask_score = extract_mask_score(adapter, top_mask, mask_config, "top window");

        auto bottom_mask_score =
                extract_mask_score(adapter_rev, bottom_mask, mask_config, "bottom window");

        BarcodeScoreResult res;
        res.adapter_name = adapter_name;
        res.kit = as.kit;
        res.top_score = top_mask_score;
        res.bottom_score = bottom_mask_score;
        res.score = std::max(res.top_score, res.bottom_score);
        res.use_top = res.top_score > res.bottom_score;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = bottom_flank_score;
        res.flank_score = res.use_top ? res.top_flank_score : res.bottom_flank_score;
        res.barcode_start = res.use_top ? top_bc_loc : bottom_start + bottom_bc_loc;
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
// of the read. So we only look for adapter sequence in the top "window" (first
// 150bp) of the read.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_adapter_score(
        std::string_view read_seq,
        const AdapterSequence& as,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_strand;
    top_strand = as.top_primer;
    int adapter_len = int(as.adapter[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_strand, read_top, adapter_len, placement_config, "top score");
    std::string_view top_mask = read_top.substr(top_bc_loc, adapter_len);
    spdlog::debug("BC location {}", top_bc_loc);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_name = as.adapter_name[i];

        if (!barcode_is_permitted(allowed_barcodes, adapter_name)) {
            continue;
        }

        spdlog::debug("Checking barcode {}", adapter_name);

        auto top_mask_score = extract_mask_score(adapter, top_mask, mask_config, "top window");

        BarcodeScoreResult res;
        res.adapter_name = adapter_name;
        res.kit = as.kit;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = -1.f;
        res.flank_score = std::max(res.top_flank_score, res.bottom_flank_score);
        res.top_score = top_mask_score;
        res.bottom_score = -1.f;
        res.score = res.top_score;
        res.use_top = true;
        res.barcode_start = top_bc_loc;
        res.top_barcode_pos = {top_result.startLocations[0], top_result.endLocations[0]};

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    return results;
}

// Score every barcode against the input read and returns the best match,
// or an unclassified match, based on certain heuristics.
BarcodeScoreResult BarcodeClassifier::find_best_adapter(
        const std::string& read_seq,
        const std::vector<AdapterSequence>& adapters,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    if (read_seq.length() < TRIM_LENGTH) {
        return UNCLASSIFIED;
    }
    std::string fwd = read_seq;

    // First find best barcode kit.
    const AdapterSequence* as;
    if (adapters.size() == 1) {
        as = &adapters[0];
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
            auto out = calculate_adapter_score_different_double_ends(fwd, *as, allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        } else {
            auto out = calculate_adapter_score_double_ends(fwd, *as, allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        }
    } else {
        auto out = calculate_adapter_score(fwd, *as, allowed_barcodes);
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
        spdlog::debug("Check double ends: top bc {}, bottom bc {}", best_top_score->adapter_name,
                      best_bottom_score->adapter_name);
        if ((best_top_score->score > 0.7) && (best_bottom_score->score > 0.7) &&
            (best_top_score->adapter_name != best_bottom_score->adapter_name)) {
            return UNCLASSIFIED;
        }
    }

    // Score the scores windows by their adapter score.
    std::sort(scores.begin(), scores.end(),
              [](const auto& l, const auto& r) { return l.score > r.score; });

    std::stringstream d;
    for (auto& s : scores) {
        d << s.score << " " << s.adapter_name << ", ";
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
