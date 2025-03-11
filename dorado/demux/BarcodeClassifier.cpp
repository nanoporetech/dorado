#include "BarcodeClassifier.h"

#include "barcoding_info.h"
#include "utils/alignment_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <edlib.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
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
            (spdlog::get_level() == spdlog::level::trace) ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;
    return mask_config;
}

// Extract the position of the barcode mask in the read based
// on the local alignment result from edlib.
int extract_mask_location(EdlibAlignResult aln, std::string_view query) {
    int query_cursor = 0;
    int target_cursor = 0;
    bool in_mask = false;
    for (int i = 0; i < aln.alignmentLength; i++) {
        if (query[query_cursor] != 'N' && in_mask) {
            break;
        }
        if (aln.alignment[i] == EDLIB_EDOP_MATCH) {
            query_cursor++;
            target_cursor++;
            if (query[query_cursor] == 'N') {
                in_mask = true;
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
    spdlog::trace("query cursor {} target cursor {}", query_cursor, target_cursor);
    return aln.startLocations[0] + target_cursor;
}

// Helper function to locally align the flanks with barcode mask
// against a subsequence of the read (either front or rear window)
// and return the alignment, score & barcode position.
std::tuple<EdlibAlignResult, float, int> extract_flank_fit(std::string_view strand,
                                                           std::string_view read,
                                                           int barcode_len,
                                                           const EdlibAlignConfig& placement_config,
                                                           const char* debug_prefix) {
    EdlibAlignResult result = edlibAlign(strand.data(), int(strand.length()), read.data(),
                                         int(read.length()), placement_config);
    float score = 1.f - static_cast<float>(result.editDistance) / (strand.length() - barcode_len);
    int bc_loc = extract_mask_location(result, strand);
    spdlog::trace("{} dist {} position {} bc_loc {} score {}", debug_prefix, result.editDistance,
                  result.startLocations[0], bc_loc, score);
    spdlog::trace("\n{}", utils::alignment_to_str(strand.data(), read.data(), result));
    return {result, score, bc_loc};
}

// Helper function to globally align a barcode to a region
// within the read.
int extract_barcode_penalty(std::string_view barcode,
                            std::string_view read,
                            const EdlibAlignConfig& config,
                            const char* debug_prefix) {
    auto result = edlibAlign(barcode.data(), int(barcode.length()), read.data(), int(read.length()),
                             config);
    auto penalty = result.editDistance;
    spdlog::trace("{} {}", debug_prefix, penalty);
    spdlog::trace("\n{}", utils::alignment_to_str(barcode.data(), read.data(), result));
    edlibFreeAlignResult(result);
    return penalty;
}

bool barcode_is_permitted(const BarcodeFilterSet& allowed_barcodes,
                          const std::string& barcode_name) {
    if (!allowed_barcodes.has_value()) {
        return true;
    }

    auto normalized_barcode_name = barcode_kits::normalize_barcode_name(barcode_name);
    return allowed_barcodes->count(normalized_barcode_name) != 0;
}

// Helper to extract left buffer from a flank.
std::string extract_left_buffer(const std::string& flank, int buffer) {
    return flank.substr(std::max(0, static_cast<int>(flank.length()) - buffer));
}

std::string extract_right_buffer(const std::string& flank, int buffer) {
    return flank.substr(0, buffer);
}

// Helper to pick the top or bottom window in a barcode. The one
// with lower penalty and higher flank score is preferred. If both
// are not satisfied by one of the windows, then just decide based
// on the barcode penalty.
std::tuple<bool, int, float> pick_top_or_bottom(int top_penalty,
                                                float top_flank_score,
                                                int bottom_penalty,
                                                float bottom_flank_score) {
    if (top_penalty <= bottom_penalty && top_flank_score >= bottom_flank_score) {
        return {true, top_penalty, top_flank_score};
    } else if (bottom_penalty <= top_penalty && bottom_flank_score >= top_flank_score) {
        return {false, bottom_penalty, bottom_flank_score};
    } else if (top_penalty <= bottom_penalty) {
        return {true, top_penalty, top_flank_score};
    } else {
        return {false, bottom_penalty, bottom_flank_score};
    }
}

}  // namespace

namespace demux {

const BarcodeScoreResult UNCLASSIFIED{};

struct BarcodeClassifier::BarcodeCandidateKit {
    std::vector<std::string> barcodes1;
    std::vector<std::string> barcodes1_rev;
    std::vector<std::string> barcodes2;
    std::vector<std::string> barcodes2_rev;
    std::string top_context;
    std::string top_context_left_buffer;
    std::string top_context_right_buffer;
    std::string top_context_rev;
    std::string top_context_rev_left_buffer;
    std::string top_context_rev_right_buffer;
    std::string bottom_context;
    std::string bottom_context_left_buffer;
    std::string bottom_context_right_buffer;
    std::string bottom_context_rev;
    std::string bottom_context_rev_left_buffer;
    std::string bottom_context_rev_right_buffer;
    std::vector<std::string> barcode_names;
    // This is the specific barcode kit product name
    // that is selected by the user, such as SQK-RBK114-96
    // or EXP-PBC096
    std::string kit;
    // This is the barcode ligation group name, such as RAB
    // or 16S, which is shared by multiple product names.
    std::string barcode_kit;
};

BarcodeClassifier::BarcodeClassifier(const std::string& kit_name)
        : m_kit_info_provider(kit_name),
          m_scoring_params(m_kit_info_provider.get_kit_info(kit_name).scoring_params),
          m_barcode_candidates(generate_candidates()) {}

BarcodeClassifier::~BarcodeClassifier() = default;

BarcodeScoreResult BarcodeClassifier::barcode(const std::string& seq,
                                              bool barcode_both_ends,
                                              const BarcodeFilterSet& allowed_barcodes) const {
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
std::vector<BarcodeClassifier::BarcodeCandidateKit> BarcodeClassifier::generate_candidates() {
    std::vector<BarcodeCandidateKit> candidates_list;

    for (auto& kit_name : m_kit_info_provider.kit_names()) {
        const barcode_kits::KitInfo& kit_info = m_kit_info_provider.get_kit_info(kit_name);

        if (!kit_info.barcodes2.empty() && kit_info.barcodes.size() != kit_info.barcodes2.size()) {
            throw std::runtime_error(
                    "If a kit has front and rear barcodes, there should be "
                    "an equal number of them");
        }

        // For click chemistry based, the flank region placement is better if the leading
        // flank sequence of the top barcode is ignored.
        bool use_leading_flank = true;
        if (kit_name.find("SQK-RBK114") != std::string::npos) {
            use_leading_flank = false;
        }

        BarcodeCandidateKit candidate;
        candidate.kit = kit_name;
        candidate.barcode_kit = kit_info.name;
        const auto& ref_bc_name = kit_info.barcodes[0];
        const std::string& ref_bc = m_kit_info_provider.get_barcode_sequence(ref_bc_name);

        std::string bc_mask(ref_bc.length(), 'N');

        // Pre-populate the sequences representing the front and rear flanks of the barcode. This
        // is generated for both ends of the barcode in double ended barcodes.
        // In addition to the flanks, a short padding sequence is also extracted from the flanks on
        // either end of the barcode. This padding sequence is used during alignment ofthe extracted mask
        // region with candidate barcodes. e.g.
        // | FLANK1 |  BC | FLANK2 |
        // | ACTGCA | CCC | GGTCAT |
        // If the padding width is 2, then instead of matching "CCC" to the extracted mask region,
        // "CACCCGG" is matched against a padded mask region. This helps anchor the front
        // and rear of the barcode flanks, and improves barcode matching.
        candidate.top_context = (use_leading_flank ? kit_info.top_front_flank : "") + bc_mask +
                                kit_info.top_rear_flank;
        candidate.top_context_left_buffer =
                extract_left_buffer(kit_info.top_front_flank, m_scoring_params.flank_left_pad);
        candidate.top_context_right_buffer =
                extract_right_buffer(kit_info.top_rear_flank, m_scoring_params.flank_right_pad);

        auto top_front_flank_rc = utils::reverse_complement(kit_info.top_front_flank);
        auto top_rear_flank_rc = utils::reverse_complement(kit_info.top_rear_flank);
        candidate.top_context_rev =
                std::string(top_rear_flank_rc).append(bc_mask).append(top_front_flank_rc);
        candidate.top_context_rev_left_buffer =
                extract_left_buffer(top_rear_flank_rc, m_scoring_params.flank_left_pad);
        candidate.top_context_rev_right_buffer =
                extract_right_buffer(top_front_flank_rc, m_scoring_params.flank_right_pad);

        if (!kit_info.barcodes2.empty()) {
            const auto& ref_bc2_name = kit_info.barcodes2[0];
            const std::string& ref_bc2 = m_kit_info_provider.get_barcode_sequence(ref_bc2_name);

            std::string bc2_mask(ref_bc2.length(), 'N');
            candidate.bottom_context = (use_leading_flank ? kit_info.bottom_front_flank : "") +
                                       bc2_mask + kit_info.bottom_rear_flank;
            candidate.bottom_context_left_buffer = extract_left_buffer(
                    kit_info.bottom_front_flank, m_scoring_params.flank_left_pad);
            candidate.bottom_context_right_buffer = extract_right_buffer(
                    kit_info.bottom_rear_flank, m_scoring_params.flank_right_pad);

            auto bottom_front_flank_rc = utils::reverse_complement(kit_info.bottom_front_flank);
            auto bottom_rear_flank_rc = utils::reverse_complement(kit_info.bottom_rear_flank);
            candidate.bottom_context_rev =
                    std::string(bottom_rear_flank_rc).append(bc_mask).append(bottom_front_flank_rc);
            candidate.bottom_context_rev_left_buffer =
                    extract_left_buffer(bottom_rear_flank_rc, m_scoring_params.flank_left_pad);
            candidate.bottom_context_rev_right_buffer =
                    extract_right_buffer(bottom_front_flank_rc, m_scoring_params.flank_right_pad);
        }

        for (size_t idx = 0; idx < kit_info.barcodes.size(); idx++) {
            const auto& bc_name = kit_info.barcodes[idx];
            const std::string& barcode1 = m_kit_info_provider.get_barcode_sequence(bc_name);
            auto barcode1_rev = utils::reverse_complement(barcode1);

            if (!candidate.barcodes1.empty() &&
                barcode1.length() != candidate.barcodes1.back().length()) {
                throw std::runtime_error(
                        "All front window barcodes must be the same length. Length for " + bc_name +
                        " is different.");
            }

            candidate.barcodes1.push_back(barcode1);
            candidate.barcodes1_rev.push_back(std::move(barcode1_rev));

            if (!kit_info.barcodes2.empty()) {
                const auto& bc2_name = kit_info.barcodes2[idx];
                const std::string& barcode2 = m_kit_info_provider.get_barcode_sequence(bc2_name);
                auto barcode2_rev = utils::reverse_complement(barcode2);

                if (!candidate.barcodes2.empty() &&
                    barcode2.length() != candidate.barcodes2.back().length()) {
                    throw std::runtime_error(
                            "All front window barcodes must be the same length. Length for " +
                            bc2_name + " is different.");
                }

                candidate.barcodes2.push_back(barcode2);
                candidate.barcodes2_rev.push_back(std::move(barcode2_rev));
            }

            candidate.barcode_names.push_back(bc_name);
        }

        candidates_list.push_back(std::move(candidate));
    }
    spdlog::debug("> Kits to evaluate: {}", candidates_list.size());
    return candidates_list;
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
// of the read. The flank sequence is also different for top and bottom contexts.
// So we need to check both ends of the read. Since the barcodes always ligate to
// 5' end of the read, the 3' end of the other strand has the reverse complement
// of that barcode sequence. This leads to 2 variants of the barcode arrangements.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score_different_double_ends(
        std::string_view read_seq,
        const BarcodeCandidateKit& candidate,
        const BarcodeFilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, m_scoring_params.front_barcode_window);
    int bottom_start =
            std::max(0, static_cast<int>(read_seq.length()) - m_scoring_params.rear_barcode_window);
    std::string_view read_bottom =
            read_seq.substr(bottom_start, m_scoring_params.rear_barcode_window);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_context_v1 = candidate.top_context;
    const auto& top_context_v1_left_buffer = candidate.top_context_left_buffer;
    const auto& top_context_v1_right_buffer = candidate.top_context_right_buffer;

    std::string_view bottom_context_v1 = candidate.bottom_context_rev;
    const auto& bottom_context_v1_left_buffer = candidate.bottom_context_rev_left_buffer;
    const auto& bottom_context_v1_right_buffer = candidate.bottom_context_rev_right_buffer;

    std::string_view top_context_v2 = candidate.bottom_context;
    const auto& top_context_v2_left_buffer = candidate.bottom_context_left_buffer;
    const auto& top_context_v2_right_buffer = candidate.bottom_context_right_buffer;

    std::string_view bottom_context_v2 = candidate.top_context_rev;
    const auto& bottom_context_v2_left_buffer = candidate.top_context_rev_left_buffer;
    const auto& bottom_context_v2_right_buffer = candidate.top_context_rev_right_buffer;

    int barcode_len = int(candidate.barcodes1[0].length());

    // Fetch barcode mask locations for variant 1
    auto [top_result_v1, top_flank_score_v1, top_bc_loc_v1] = extract_flank_fit(
            top_context_v1, read_top, barcode_len, placement_config, "top score v1");
    auto top_start_idx_v1 = std::max(
            0, top_bc_loc_v1 - static_cast<int>(top_context_v1_left_buffer.length()) - barcode_len);
    auto top_end_idx_v1 = top_bc_loc_v1 + static_cast<int>(top_context_v1_right_buffer.length());
    std::string_view top_mask_v1 =
            read_top.substr(top_start_idx_v1, top_end_idx_v1 - top_start_idx_v1);

    auto [bottom_result_v1, bottom_flank_score_v1, bottom_bc_loc_v1] = extract_flank_fit(
            bottom_context_v1, read_bottom, barcode_len, placement_config, "bottom score v1");
    auto bottom_start_idx_v1 = std::max(
            0, bottom_bc_loc_v1 - static_cast<int>(bottom_context_v1_left_buffer.length()) -
                       barcode_len);
    auto bottom_end_idx_v1 =
            bottom_bc_loc_v1 + static_cast<int>(bottom_context_v1_right_buffer.length());
    std::string_view bottom_mask_v1 =
            read_bottom.substr(bottom_start_idx_v1, bottom_end_idx_v1 - bottom_start_idx_v1);

    // Fetch barcode mask locations for variant 2
    auto [top_result_v2, top_flank_score_v2, top_bc_loc_v2] = extract_flank_fit(
            top_context_v2, read_top, barcode_len, placement_config, "top score v2");
    auto top_start_idx_v2 = std::max(
            0, top_bc_loc_v2 - static_cast<int>(top_context_v2_left_buffer.length()) - barcode_len);
    auto top_end_idx_v2 = top_bc_loc_v2 + static_cast<int>(top_context_v2_right_buffer.length());
    std::string_view top_mask_v2 =
            read_top.substr(top_start_idx_v2, top_end_idx_v2 - top_start_idx_v2);

    auto [bottom_result_v2, bottom_flank_score_v2, bottom_bc_loc_v2] = extract_flank_fit(
            bottom_context_v2, read_bottom, barcode_len, placement_config, "bottom score v2");
    auto bottom_start_idx_v2 = std::max(
            0, bottom_bc_loc_v2 - static_cast<int>(bottom_context_v2_left_buffer.length()) -
                       barcode_len);
    auto bottom_end_idx_v2 =
            bottom_bc_loc_v2 + static_cast<int>(bottom_context_v2_right_buffer.length());
    std::string_view bottom_mask_v2 =
            read_bottom.substr(bottom_start_idx_v2, bottom_end_idx_v2 - bottom_start_idx_v2);

    // Find the best variant of the two.
    int total_v1_penalty = top_result_v1.editDistance + bottom_result_v1.editDistance;
    int total_v2_penalty = top_result_v2.editDistance + bottom_result_v2.editDistance;
    spdlog::trace("total v1 edit dist {}, total v2 edit dis {}", total_v1_penalty,
                  total_v2_penalty);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < candidate.barcodes1.size(); i++) {
        const auto barcode1 = std::string(top_context_v1_left_buffer)
                                      .append(candidate.barcodes1[i])
                                      .append(top_context_v1_right_buffer);
        const auto barcode1_rev = std::string(bottom_context_v2_left_buffer)
                                          .append(candidate.barcodes1_rev[i])
                                          .append(bottom_context_v2_right_buffer);
        const auto barcode2 = std::string(top_context_v2_left_buffer)
                                      .append(candidate.barcodes2[i])
                                      .append(top_context_v2_right_buffer);
        const auto barcode2_rev = std::string(bottom_context_v1_left_buffer)
                                          .append(candidate.barcodes2_rev[i])
                                          .append(bottom_context_v1_right_buffer);
        auto& barcode_name = candidate.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }

        spdlog::trace("Checking barcode {}", barcode_name);

        // Calculate barcode penalties for v1.
        auto top_mask_result_penalty_v1 =
                extract_barcode_penalty(barcode1, top_mask_v1, mask_config, "top window v1");

        auto bottom_mask_result_penalty_v1 = extract_barcode_penalty(
                barcode2_rev, bottom_mask_v1, mask_config, "bottom window v1");

        BarcodeScoreResult v1;
        v1.top_penalty = top_mask_result_penalty_v1;
        v1.bottom_penalty = bottom_mask_result_penalty_v1;
        v1.top_flank_score = top_flank_score_v1;
        v1.bottom_flank_score = bottom_flank_score_v1;
        std::tie(v1.use_top, v1.penalty, v1.flank_score) = pick_top_or_bottom(
                v1.top_penalty, v1.top_flank_score, v1.bottom_penalty, v1.bottom_flank_score);
        v1.top_barcode_score = (1.f - static_cast<float>(v1.top_penalty) / barcode1.length());
        v1.bottom_barcode_score =
                (1.f - static_cast<float>(v1.bottom_penalty) / barcode2_rev.length());
        v1.barcode_score = v1.use_top ? v1.top_barcode_score : v1.bottom_barcode_score;
        v1.top_barcode_pos = {top_result_v1.startLocations[0], top_result_v1.endLocations[0]};
        v1.bottom_barcode_pos = {bottom_start + bottom_result_v1.startLocations[0],
                                 bottom_start + bottom_result_v1.endLocations[0]};

        // Calculate barcode penalties for v2.
        auto top_mask_result_penalty_v2 =
                extract_barcode_penalty(barcode2, top_mask_v2, mask_config, "top window v2");

        auto bottom_mask_result_penalty_v2 = extract_barcode_penalty(
                barcode1_rev, bottom_mask_v2, mask_config, "bottom window v2");

        BarcodeScoreResult v2;
        v2.top_penalty = top_mask_result_penalty_v2;
        v2.bottom_penalty = bottom_mask_result_penalty_v2;
        v2.top_flank_score = top_flank_score_v2;
        v2.bottom_flank_score = bottom_flank_score_v2;
        std::tie(v2.use_top, v2.penalty, v2.flank_score) = pick_top_or_bottom(
                v2.top_penalty, v2.top_flank_score, v2.bottom_penalty, v2.bottom_flank_score);
        v2.top_barcode_score = (1.f - static_cast<float>(v2.top_penalty) / barcode2.length());
        v2.bottom_barcode_score =
                (1.f - static_cast<float>(v2.bottom_penalty) / barcode1_rev.length());
        v2.barcode_score = v2.use_top ? v2.top_barcode_score : v2.bottom_barcode_score;
        v2.top_barcode_pos = {top_result_v2.startLocations[0], top_result_v2.endLocations[0]};
        v2.bottom_barcode_pos = {bottom_start + bottom_result_v2.startLocations[0],
                                 bottom_start + bottom_result_v2.endLocations[0]};

        // The best variant is the one with lower penalty for both barcode
        // and flanks. If that's not clear, then just use the barcode score
        // penalty to decide.
        bool var1_is_best = true;
        if (v1.penalty <= v2.penalty && total_v1_penalty <= total_v2_penalty) {
            var1_is_best = true;
        } else if (v2.penalty <= v1.penalty && total_v2_penalty <= total_v1_penalty) {
            var1_is_best = false;
        } else if (v1.penalty <= v2.penalty) {
            var1_is_best = true;
        } else {
            var1_is_best = false;
        }
        BarcodeScoreResult res = var1_is_best ? v1 : v2;
        res.variant = var1_is_best ? "var1" : "var2";
        res.barcode_name = barcode_name;
        res.kit = candidate.kit;
        res.barcode_kit = candidate.barcode_kit;

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result_v1);
    edlibFreeAlignResult(bottom_result_v1);
    edlibFreeAlignResult(top_result_v2);
    edlibFreeAlignResult(bottom_result_v2);
    return results;
}

float BarcodeClassifier::find_midstrand_barcode_different_double_ends(
        std::string_view read_seq,
        const BarcodeCandidateKit& candidate) const {
    auto length_of_end_windows =
            m_scoring_params.front_barcode_window + m_scoring_params.rear_barcode_window;
    if ((int)read_seq.length() < length_of_end_windows) {
        return 0.f;
    }

    std::string_view top_context_v1 = candidate.top_context;
    std::string_view bottom_context_v1 = candidate.bottom_context_rev;

    std::string_view top_context_v2 = candidate.bottom_context;
    std::string_view bottom_context_v2 = candidate.top_context_rev;

    auto length_without_end_windows = read_seq.length() - length_of_end_windows;

    if (length_without_end_windows <
        std::min({top_context_v1.length(), bottom_context_v1.length(), top_context_v2.length(),
                  bottom_context_v2.length()})) {
        return 0.f;
    }

    auto read_mid =
            read_seq.substr(m_scoring_params.front_barcode_window, length_without_end_windows);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    int barcode_len = int(candidate.barcodes1[0].length());

    // Fetch barcode mask locations for variant 1
    auto [top_result_v1, top_flank_score_v1, top_bc_loc_v1] = extract_flank_fit(
            top_context_v1, read_mid, barcode_len, placement_config, "midstrand flank top v1");

    auto [bottom_result_v1, bottom_flank_score_v1, bottom_bc_loc_v1] =
            extract_flank_fit(bottom_context_v1, read_mid, barcode_len, placement_config,
                              "midstrand flank bottom v1");

    // Fetch barcode mask locations for variant 2
    auto [top_result_v2, top_flank_score_v2, top_bc_loc_v2] = extract_flank_fit(
            top_context_v2, read_mid, barcode_len, placement_config, "midstrand flank top v2");

    auto [bottom_result_v2, bottom_flank_score_v2, bottom_bc_loc_v2] =
            extract_flank_fit(bottom_context_v2, read_mid, barcode_len, placement_config,
                              "midstrand flank bottom v2");

    edlibFreeAlignResult(top_result_v1);
    edlibFreeAlignResult(bottom_result_v1);
    edlibFreeAlignResult(top_result_v2);
    edlibFreeAlignResult(bottom_result_v2);
    // Find the best variant of the two.
    return std::max(
            {top_flank_score_v1, bottom_flank_score_v1, top_flank_score_v2, bottom_flank_score_v2});
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
// same for top and bottom contexts, we simply need to look for the barcode and its
// reverse complement sequence in the top/bottom windows.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score_double_ends(
        std::string_view read_seq,
        const BarcodeCandidateKit& candidate,
        const BarcodeFilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, m_scoring_params.front_barcode_window);
    int bottom_start =
            std::max(0, static_cast<int>(read_seq.length()) - m_scoring_params.rear_barcode_window);
    std::string_view read_bottom =
            read_seq.substr(bottom_start, m_scoring_params.rear_barcode_window);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_context = candidate.top_context;
    const auto& top_left_buffer = candidate.top_context_left_buffer;
    const auto& top_right_buffer = candidate.top_context_right_buffer;

    std::string_view bottom_context = candidate.top_context_rev;
    const auto& bottom_left_buffer = candidate.top_context_rev_left_buffer;
    const auto& bottom_right_buffer = candidate.top_context_rev_right_buffer;

    int barcode_len = int(candidate.barcodes1[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_context, read_top, barcode_len, placement_config, "top score");
    auto top_start_idx =
            std::max(0, top_bc_loc - static_cast<int>(top_left_buffer.length()) - barcode_len);
    auto top_end_idx = top_bc_loc + static_cast<int>(top_right_buffer.length());
    std::string_view top_mask = read_top.substr(top_start_idx, top_end_idx - top_start_idx);

    auto [bottom_result, bottom_flank_score, bottom_bc_loc] = extract_flank_fit(
            bottom_context, read_bottom, barcode_len, placement_config, "bottom score");
    auto bottom_start_idx = std::max(
            0, bottom_bc_loc - static_cast<int>(bottom_left_buffer.length()) - barcode_len);
    auto bottom_end_idx = bottom_bc_loc + static_cast<int>(bottom_right_buffer.length());
    std::string_view bottom_mask =
            read_bottom.substr(bottom_start_idx, bottom_end_idx - bottom_start_idx);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < candidate.barcodes1.size(); i++) {
        auto barcode = std::string(top_left_buffer)
                               .append(candidate.barcodes1[i])
                               .append(top_right_buffer);
        auto barcode_rev = std::string(bottom_left_buffer)
                                   .append(candidate.barcodes1_rev[i])
                                   .append(bottom_right_buffer);
        auto& barcode_name = candidate.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }
        spdlog::trace("Checking barcode {}", barcode_name);

        auto top_mask_penalty =
                extract_barcode_penalty(barcode, top_mask, mask_config, "top window");

        auto bottom_mask_penalty =
                extract_barcode_penalty(barcode_rev, bottom_mask, mask_config, "bottom window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = candidate.kit;
        res.barcode_kit = candidate.barcode_kit;
        res.top_penalty = top_mask_penalty;
        res.bottom_penalty = bottom_mask_penalty;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = bottom_flank_score;
        std::tie(res.use_top, res.penalty, res.flank_score) = pick_top_or_bottom(
                res.top_penalty, res.top_flank_score, res.bottom_penalty, res.bottom_flank_score);
        res.top_barcode_score = (1.f - static_cast<float>(res.top_penalty) / barcode.length());
        res.bottom_barcode_score =
                (1.f - static_cast<float>(res.bottom_penalty) / barcode_rev.length());
        res.barcode_score = res.use_top ? res.top_barcode_score : res.bottom_barcode_score;
        res.top_barcode_pos = {top_result.startLocations[0], top_result.endLocations[0]};
        res.bottom_barcode_pos = {bottom_start + bottom_result.startLocations[0],
                                  bottom_start + bottom_result.endLocations[0]};

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    edlibFreeAlignResult(bottom_result);
    return results;
}

float BarcodeClassifier::find_midstrand_barcode_double_ends(
        std::string_view read_seq,
        const BarcodeCandidateKit& candidate) const {
    auto length_of_end_windows =
            m_scoring_params.front_barcode_window + m_scoring_params.rear_barcode_window;
    if ((int)read_seq.length() < length_of_end_windows) {
        return 0.f;
    }

    std::string_view top_context = candidate.top_context;
    std::string_view bottom_context = candidate.top_context_rev;

    auto length_without_end_windows = read_seq.length() - length_of_end_windows;

    if (length_without_end_windows < std::min({top_context.length(), bottom_context.length()})) {
        return 0.f;
    }

    auto read_mid =
            read_seq.substr(m_scoring_params.front_barcode_window, length_without_end_windows);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    int barcode_len = int(candidate.barcodes1[0].length());

    auto [top_result, top_flank_score, top_bc_loc] = extract_flank_fit(
            top_context, read_mid, barcode_len, placement_config, "midstrand flank top");

    auto [bottom_result, bottom_flank_score, bottom_bc_loc] = extract_flank_fit(
            bottom_context, read_mid, barcode_len, placement_config, "midstrand flank bottom");

    edlibFreeAlignResult(top_result);
    edlibFreeAlignResult(bottom_result);
    // Find the best variant of the two.
    return std::max({top_flank_score, bottom_flank_score});
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====---------------> 3'
//      BCXXX
//
// In this scenario, the barcode (and its flanks) only ligate to the 5' end
// of the read. So we only look for barcode sequence in the top "window" (first
// 150bp) of the read.
// If rear_barcodes is true, the barcode ligates only at the 3' end, so we shift the
// window to the end of the read.
std::vector<BarcodeScoreResult> BarcodeClassifier::calculate_barcode_score(
        std::string_view read_seq,
        const BarcodeCandidateKit& candidate,
        const BarcodeFilterSet& allowed_barcodes,
        bool rear_barcodes) const {
    std::string_view read_top;
    if (rear_barcodes) {
        int rear_start = std::max(0, (int)read_seq.length() - m_scoring_params.rear_barcode_window);
        read_top = read_seq.substr(rear_start, m_scoring_params.rear_barcode_window);
    } else {
        read_top = read_seq.substr(0, m_scoring_params.front_barcode_window);
    }

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_context = candidate.top_context;
    int barcode_len = int(candidate.barcodes1[0].length());
    const auto& top_left_buffer = candidate.top_context_left_buffer;
    const auto& top_right_buffer = candidate.top_context_right_buffer;

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_context, read_top, barcode_len, placement_config, "top score");
    auto start_idx =
            std::max(0, top_bc_loc - static_cast<int>(top_left_buffer.length()) - barcode_len);
    auto end_idx = top_bc_loc + static_cast<int>(top_right_buffer.length());
    std::string_view top_mask = read_top.substr(start_idx, end_idx - start_idx);

    spdlog::trace("BC location {}", top_bc_loc);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < candidate.barcodes1.size(); i++) {
        const auto barcode = candidate.top_context_left_buffer + candidate.barcodes1[i] +
                             candidate.top_context_right_buffer;
        auto& barcode_name = candidate.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }
        spdlog::trace("Checking barcode {}", barcode_name);

        auto top_mask_penalty =
                extract_barcode_penalty(barcode, top_mask, mask_config, "top window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = candidate.kit;
        res.barcode_kit = candidate.barcode_kit;
        if (rear_barcodes) {
            res.bottom_flank_score = top_flank_score;
            res.flank_score = res.bottom_flank_score;
            res.bottom_penalty = top_mask_penalty;
            res.penalty = res.bottom_penalty;
            res.use_top = false;
            res.bottom_barcode_score =
                    1.f - static_cast<float>(res.bottom_penalty) / barcode.length();
            res.barcode_score = res.bottom_barcode_score;
            int rear_start =
                    std::max(0, (int)read_seq.length() - m_scoring_params.rear_barcode_window);
            res.bottom_barcode_pos = {rear_start + top_result.startLocations[0],
                                      rear_start + top_result.endLocations[0]};
        } else {
            res.top_flank_score = top_flank_score;
            res.flank_score = res.top_flank_score;
            res.top_penalty = top_mask_penalty;
            res.penalty = res.top_penalty;
            res.use_top = true;
            res.top_barcode_score = 1.f - static_cast<float>(res.top_penalty) / barcode.length();
            res.barcode_score = res.top_barcode_score;
            res.top_barcode_pos = {top_result.startLocations[0], top_result.endLocations[0]};
        }
        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    return results;
}

float BarcodeClassifier::find_midstrand_barcode_single_end(std::string_view read_seq,
                                                           const BarcodeCandidateKit& candidate,
                                                           bool rear_barcodes) const {
    auto length_of_end_window = rear_barcodes ? m_scoring_params.rear_barcode_window
                                              : m_scoring_params.front_barcode_window;
    if ((int)read_seq.length() < length_of_end_window) {
        return 0.f;
    }

    std::string_view top_context = candidate.top_context;

    auto length_without_end_window = read_seq.length() - length_of_end_window;

    if (length_without_end_window < top_context.length()) {
        return 0.f;
    }

    auto read_mid = rear_barcodes ? read_seq.substr(0, length_without_end_window)
                                  : read_seq.substr(m_scoring_params.front_barcode_window,
                                                    length_without_end_window);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    int barcode_len = int(candidate.barcodes1[0].length());

    auto [top_result, top_flank_score, top_bc_loc] = extract_flank_fit(
            top_context, read_mid, barcode_len, placement_config, "midstrand flank top");

    edlibFreeAlignResult(top_result);

    return top_flank_score;
}

// Score every barcode against the input read and returns the best match,
// or an unclassified match, based on certain heuristics.
BarcodeScoreResult BarcodeClassifier::find_best_barcode(
        const std::string& read_seq,
        const std::vector<BarcodeCandidateKit>& candidates,
        bool barcode_both_ends,
        const BarcodeFilterSet& allowed_barcodes) const {
    if (read_seq.length() == 0) {
        return UNCLASSIFIED;
    }

    const std::string_view fwd = read_seq;

    // First find best barcode kit.
    const BarcodeCandidateKit* candidate;
    if (candidates.size() == 1) {
        candidate = &candidates[0];
    } else {
        // TODO: Implement finding best kit match.
        throw std::runtime_error("Unimplemented: multiple barcoding kits");
    }

    const barcode_kits::KitInfo& kit = m_kit_info_provider.get_kit_info(candidate->kit);

    // Detect presence of mid-strand barcode. If one is confident found, then
    // treat that read as unclassified since it's most likely an unsplit read.
    float midstrand_score = -1.f;
    if (kit.double_ends) {
        if (kit.ends_different) {
            midstrand_score = find_midstrand_barcode_different_double_ends(fwd, *candidate);
        } else {
            midstrand_score = find_midstrand_barcode_double_ends(fwd, *candidate);
        }
    } else {
        midstrand_score =
                find_midstrand_barcode_single_end(fwd, *candidate, kit.rear_only_barcodes);
    }
    const auto midstrand_thres = m_scoring_params.midstrand_flank_score;
    if (midstrand_score >= midstrand_thres) {
        spdlog::trace("Found midstrand barcode flanks with score {}, threshold {}", midstrand_score,
                      midstrand_thres);
        auto midstrand_res = UNCLASSIFIED;
        midstrand_res.found_midstrand = true;
        return midstrand_res;
    }

    // Then find the best barcode hit within that kit.
    std::vector<BarcodeScoreResult> results;
    if (kit.double_ends) {
        if (kit.ends_different) {
            auto out = calculate_barcode_score_different_double_ends(fwd, *candidate,
                                                                     allowed_barcodes);
            results.insert(results.end(), out.begin(), out.end());
        } else {
            auto out = calculate_barcode_score_double_ends(fwd, *candidate, allowed_barcodes);
            results.insert(results.end(), out.begin(), out.end());
        }
    } else {
        auto out =
                calculate_barcode_score(fwd, *candidate, allowed_barcodes, kit.rear_only_barcodes);
        results.insert(results.end(), out.begin(), out.end());
    }

    if (results.empty()) {
        spdlog::warn("Barcode unclassified because no barcodes found in kit.");
        return UNCLASSIFIED;
    }

    // Sort the scores windows by their barcode score.
    std::sort(results.begin(), results.end(),
              [](const auto& l, const auto& r) { return l.penalty < r.penalty; });

    std::stringstream d;
    for (auto& s : results) {
        d << s.barcode_name << " " << s.penalty << ", ";
    }
    spdlog::trace("Scores: {}", d.str());
    auto best_result = results.begin();
    auto are_penalties_acceptable = [this](const auto& proposal) {
        // If barcode penalty is 0, it's a perfect match. Consider it a pass.
        return (proposal.penalty == 0) ||
               ((proposal.penalty <= m_scoring_params.max_barcode_penalty) &&
                (proposal.flank_score >= m_scoring_params.min_flank_score));
    };

    BarcodeScoreResult out = UNCLASSIFIED;
    if (results.size() == 1) {
        if (are_penalties_acceptable(*best_result)) {
            out = *best_result;
        }
    } else {
        const auto& second_best_result = std::next(best_result);
        const int penalty_dist = second_best_result->penalty - best_result->penalty;

        // NB: this test will pass if _either_ barcode is close enough to the end of the read for a both ends kit.
        bool barcode_proximity_ok =
                (best_result->top_barcode_pos.first >= 0 &&
                 best_result->top_barcode_pos.first <= m_scoring_params.barcode_end_proximity) ||
                (best_result->bottom_barcode_pos.second >= 0 &&
                 best_result->bottom_barcode_pos.second >=
                         int(read_seq.length() - m_scoring_params.barcode_end_proximity));

        if (((penalty_dist >= m_scoring_params.min_barcode_penalty_dist &&
              are_penalties_acceptable(*best_result)) ||
             (penalty_dist >= m_scoring_params.min_separation_only_dist)) &&
            barcode_proximity_ok) {
            out = *best_result;
        }
    }

    if (barcode_both_ends && kit.double_ends) {
        // For more stringent classification, ensure that both ends of a read
        // have a high score for the same barcode. If not then consider it
        // unclassified.
        if (std::max(out.top_penalty, out.bottom_penalty) > m_scoring_params.max_barcode_penalty) {
            spdlog::trace("Max of top {} and bottom penalties {} > max barcode penalty {}",
                          out.top_penalty, out.bottom_penalty,
                          m_scoring_params.max_barcode_penalty);
            return UNCLASSIFIED;
        }
    }

    if (kit.double_ends) {
        // For more stringent classification, ensure that neither end of a read has a higher scoring
        //  barcode, if any of the barcodes at that end are better than the threshold.
        auto best_top_result = std::min_element(
                results.begin(), results.end(),
                [](const auto& l, const auto& r) { return l.top_penalty < r.top_penalty; });
        auto best_bottom_result = std::min_element(
                results.begin(), results.end(),
                [](const auto& l, const auto& r) { return l.bottom_penalty < r.bottom_penalty; });

        if (out.barcode_name != UNCLASSIFIED.barcode_name &&
            (((out.barcode_name != best_top_result->barcode_name) &&
              (best_top_result->top_penalty <= m_scoring_params.max_barcode_penalty)) ||
             ((out.barcode_name != best_bottom_result->barcode_name) &&
              (best_bottom_result->bottom_penalty <= m_scoring_params.max_barcode_penalty)))) {
            spdlog::trace(
                    "Superior barcode found for arrangement {} : top best bc {}, bottom best bc {}",
                    out.barcode_name, best_top_result->barcode_name,
                    best_bottom_result->barcode_name);
            return UNCLASSIFIED;
        }
    }

    // If nothing is found, report as unclassified.
    return out;
}

}  // namespace demux

}  // namespace dorado
