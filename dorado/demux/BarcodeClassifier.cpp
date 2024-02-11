#include "BarcodeClassifier.h"

#include "parse_custom_kit.h"
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
float extract_mask_score(std::string_view barcode,
                         std::string_view read,
                         const EdlibAlignConfig& config,
                         const char* debug_prefix) {
    auto result = edlibAlign(barcode.data(), int(barcode.length()), read.data(), int(read.length()),
                             config);
    auto score = result.editDistance;
    spdlog::trace("{} {} position {}", debug_prefix, score, result.startLocations[0]);
    spdlog::trace("\n{}", utils::alignment_to_str(barcode.data(), read.data(), result));
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

// Helper function to convert the parsed custom kit tuple
// into an unordered_map to simplify searching for kit info during
// barcoding.
std::unordered_map<std::string, dorado::barcode_kits::KitInfo> process_custom_kit(
        const std::optional<std::string>& custom_kit) {
    std::unordered_map<std::string, dorado::barcode_kits::KitInfo> kit_map;
    if (custom_kit) {
        auto custom_arrangement = demux::parse_custom_arrangement(*custom_kit);
        if (custom_arrangement) {
            const auto& [kit_name, kit_info] = *custom_arrangement;
            kit_map[kit_name] = kit_info;
        }
    }
    return kit_map;
}

// Helper to extract left buffer from a flank.
std::string extract_left_buffer(const std::string& flank, int buffer) {
    return flank.substr(std::max(0lu, flank.length() - buffer));
}

std::string extract_right_buffer(const std::string& flank, int buffer) {
    ;
    return flank.substr(0, buffer);
}

}  // namespace

namespace demux {

const int TRIM_LENGTH = 175;
const int LEFT_BUFFER = 5;
const int RIGHT_BUFFER = 10;
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
    std::string kit;
    std::string barcode_kit;
};

BarcodeClassifier::BarcodeClassifier(const std::vector<std::string>& kit_names,
                                     const std::optional<std::string>& custom_kit,
                                     const std::optional<std::string>& custom_barcodes)
        : m_custom_kit(process_custom_kit(custom_kit)),
          m_custom_seqs(custom_barcodes ? parse_custom_sequences(*custom_barcodes)
                                        : std::unordered_map<std::string, std::string>{}),
          m_scoring_params(custom_kit ? parse_scoring_params(*custom_kit)
                                      : BarcodeKitScoringParams{}),
          m_barcode_candidates(generate_candidates(kit_names)) {}

BarcodeClassifier::~BarcodeClassifier() = default;

BarcodeScoreResult BarcodeClassifier::barcode(
        const std::string& seq,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    auto best_barcode =
            find_best_barcode(seq, m_barcode_candidates, barcode_both_ends, allowed_barcodes);
    return best_barcode;
}

const dorado::barcode_kits::KitInfo& BarcodeClassifier::get_kit_info(
        const std::string& kit_name) const {
    auto custom_kit_iter = m_custom_kit.find(kit_name);
    if (custom_kit_iter != m_custom_kit.end()) {
        return custom_kit_iter->second;
    }
    const auto& kit_info_map = barcode_kits::get_kit_infos();
    auto prebuilt_kit_iter = kit_info_map.find(kit_name);
    if (prebuilt_kit_iter != kit_info_map.end()) {
        return prebuilt_kit_iter->second;
    }
    throw std::runtime_error("Could not find " + kit_name + " in pre-built or custom kits");
}

const std::string& BarcodeClassifier::get_barcode_sequence(const std::string& barcode_name) const {
    auto custom_seqs_iter = m_custom_seqs.find(barcode_name);
    if (custom_seqs_iter != m_custom_seqs.end()) {
        return custom_seqs_iter->second;
    }
    const auto& barcodes = barcode_kits::get_barcodes();
    auto prebuilt_seqs_iter = barcodes.find(barcode_name);
    if (prebuilt_seqs_iter != barcodes.end()) {
        return prebuilt_seqs_iter->second;
    }
    throw std::runtime_error("Could not find " + barcode_name +
                             " in pre-built or custom barcode sequences");
}

// Generate all possible barcode candidates. If kit name is passed
// limit the candidates generated to only the specified kits. This is done
// to frontload some of the computation, such as calculating flanks
// and their reverse complements, barcode sequences and their reverse complements,
// etc.
// Returns a vector all barcode candidates to test the
// input read sequence against.
std::vector<BarcodeClassifier::BarcodeCandidateKit> BarcodeClassifier::generate_candidates(
        const std::vector<std::string>& kit_names) {
    std::vector<BarcodeCandidateKit> candidates_list;

    std::vector<std::string> final_kit_names;
    if (!m_custom_kit.empty()) {
        for (auto& [kit_name, _] : m_custom_kit) {
            final_kit_names.push_back(kit_name);
        }
    } else if (kit_names.empty()) {
        throw std::runtime_error(
                "Either custom kit must include kit arrangement or a kit name needs to be passed "
                "in.");
    } else {
        final_kit_names = kit_names;
    }

    for (auto& kit_name : final_kit_names) {
        const auto& kit_info = get_kit_info(kit_name);

        if (!kit_info.barcodes2.empty() && kit_info.barcodes.size() != kit_info.barcodes2.size()) {
            throw std::runtime_error(
                    "If a kit has front and rear barcodes, there should be "
                    "an equal number of them");
        }

        bool use_leading_flank = true;
        if (kit_name.find("SQK-RBK114") != std::string::npos) {
            use_leading_flank = false;
        }

        // Update left and right buffer lengths based on the actual flank regions of the
        // chosen kit.
        m_left_buffer =
                std::min({m_scoring_params.flank_left_pad, int(kit_info.top_front_flank.length()),
                          int(kit_info.top_rear_flank.length())});
        m_right_buffer =
                std::min({m_scoring_params.flank_right_pad, int(kit_info.top_rear_flank.length()),
                          int(kit_info.top_front_flank.length())});
        if (!kit_info.barcodes2.empty()) {
            m_left_buffer = std::min({int(m_left_buffer), int(kit_info.bottom_front_flank.length()),
                                      int(kit_info.bottom_rear_flank.length())});
            m_right_buffer =
                    std::min({int(m_right_buffer), int(kit_info.bottom_rear_flank.length()),
                              int(kit_info.bottom_front_flank.length())});
        }

        BarcodeCandidateKit candidate;
        candidate.kit = kit_name;
        candidate.barcode_kit = kit_info.name;
        const auto& ref_bc_name = kit_info.barcodes[0];
        const auto& ref_bc = get_barcode_sequence(ref_bc_name);

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
                extract_left_buffer(kit_info.top_front_flank, m_left_buffer);
        candidate.top_context_right_buffer =
                extract_right_buffer(kit_info.top_rear_flank, m_right_buffer);

        auto top_front_flank_rc = utils::reverse_complement(kit_info.top_front_flank);
        auto top_rear_flank_rc = utils::reverse_complement(kit_info.top_rear_flank);
        candidate.top_context_rev = top_rear_flank_rc + bc_mask + top_front_flank_rc;
        candidate.top_context_rev_left_buffer =
                extract_left_buffer(top_rear_flank_rc, m_left_buffer);
        candidate.top_context_rev_right_buffer =
                extract_right_buffer(top_front_flank_rc, m_right_buffer);

        if (!kit_info.barcodes2.empty()) {
            const auto& ref_bc2_name = kit_info.barcodes2[0];
            const auto& ref_bc2 = get_barcode_sequence(ref_bc2_name);

            std::string bc2_mask(ref_bc2.length(), 'N');
            candidate.bottom_context = (use_leading_flank ? kit_info.bottom_front_flank : "") +
                                       bc2_mask + kit_info.bottom_rear_flank;
            candidate.bottom_context_left_buffer =
                    extract_left_buffer(kit_info.bottom_front_flank, m_left_buffer);
            candidate.bottom_context_right_buffer =
                    extract_right_buffer(kit_info.bottom_rear_flank, m_right_buffer);

            auto bottom_front_flank_rc = utils::reverse_complement(kit_info.bottom_front_flank);
            auto bottom_rear_flank_rc = utils::reverse_complement(kit_info.bottom_rear_flank);
            candidate.bottom_context_rev = bottom_rear_flank_rc + bc_mask + bottom_front_flank_rc;
            candidate.bottom_context_rev_left_buffer =
                    extract_left_buffer(bottom_rear_flank_rc, m_left_buffer);
            candidate.bottom_context_rev_right_buffer =
                    extract_right_buffer(bottom_front_flank_rc, m_right_buffer);
        }

        for (size_t idx = 0; idx < kit_info.barcodes.size(); idx++) {
            const auto& bc_name = kit_info.barcodes[idx];
            const auto& barcode1 = get_barcode_sequence(bc_name);
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
                const auto& barcode2 = get_barcode_sequence(bc2_name);
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
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

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
    auto top_start_idx_v1 = std::max(0, top_bc_loc_v1 - m_left_buffer - barcode_len);
    auto top_end_idx_v1 = top_bc_loc_v1 + m_right_buffer;
    std::string_view top_mask_v1 =
            read_top.substr(top_start_idx_v1, top_end_idx_v1 - top_start_idx_v1);

    auto [bottom_result_v1, bottom_flank_score_v1, bottom_bc_loc_v1] = extract_flank_fit(
            bottom_context_v1, read_bottom, barcode_len, placement_config, "bottom score v1");
    auto bottom_start_idx_v1 = std::max(0, bottom_bc_loc_v1 - m_left_buffer - barcode_len);
    auto bottom_end_idx_v1 = bottom_bc_loc_v1 + m_right_buffer;
    std::string_view bottom_mask_v1 =
            read_bottom.substr(bottom_start_idx_v1, bottom_end_idx_v1 - bottom_start_idx_v1);

    // Fetch barcode mask locations for variant 2
    auto [top_result_v2, top_flank_score_v2, top_bc_loc_v2] = extract_flank_fit(
            top_context_v2, read_top, barcode_len, placement_config, "top score v2");
    auto top_start_idx_v2 = std::max(0, top_bc_loc_v2 - m_left_buffer - barcode_len);
    auto top_end_idx_v2 = top_bc_loc_v2 + m_right_buffer;
    std::string_view top_mask_v2 =
            read_top.substr(top_start_idx_v2, top_end_idx_v2 - top_start_idx_v2);

    auto [bottom_result_v2, bottom_flank_score_v2, bottom_bc_loc_v2] = extract_flank_fit(
            bottom_context_v2, read_bottom, barcode_len, placement_config, "bottom score v2");
    auto bottom_start_idx_v2 = std::max(0, bottom_bc_loc_v2 - m_left_buffer - barcode_len);
    auto bottom_end_idx_v2 = bottom_bc_loc_v2 + m_right_buffer;
    std::string_view bottom_mask_v2 =
            read_bottom.substr(bottom_start_idx_v2, bottom_end_idx_v2 - bottom_start_idx_v2);

    // Find the best variant of the two.
    int total_v1_score = top_result_v1.editDistance + bottom_result_v1.editDistance;
    int total_v2_score = top_result_v2.editDistance + bottom_result_v2.editDistance;

    std::string_view top_mask, bottom_mask;
    if (total_v1_score < total_v2_score) {
        top_mask = top_mask_v1;
        bottom_mask = bottom_mask_v1;
        spdlog::trace("best variant v1");
    } else {
        top_mask = top_mask_v2;
        bottom_mask = bottom_mask_v2;
        spdlog::trace("best variant v2");
    }

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < candidate.barcodes1.size(); i++) {
        const auto barcode1 =
                top_context_v1_left_buffer + candidate.barcodes1[i] + top_context_v1_right_buffer;
        const auto barcode1_rev = bottom_context_v2_left_buffer + candidate.barcodes1_rev[i] +
                                  bottom_context_v2_right_buffer;
        const auto barcode2 =
                top_context_v2_left_buffer + candidate.barcodes2[i] + top_context_v2_right_buffer;
        const auto barcode2_rev = bottom_context_v1_left_buffer + candidate.barcodes2_rev[i] +
                                  bottom_context_v1_right_buffer;
        auto& barcode_name = candidate.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }

        spdlog::trace("Checking barcode {}", barcode_name);

        // Calculate barcode scores for v1.
        auto top_mask_result_score_v1 =
                extract_mask_score(barcode1, top_mask_v1, mask_config, "top window v1");

        auto bottom_mask_result_score_v1 =
                extract_mask_score(barcode2_rev, bottom_mask_v1, mask_config, "bottom window v1");

        BarcodeScoreResult v1;
        v1.top_score = top_mask_result_score_v1;
        v1.bottom_score = bottom_mask_result_score_v1;
        v1.score = std::min(v1.top_score, v1.bottom_score);
        v1.use_top = v1.top_score < v1.bottom_score;
        v1.top_flank_score = top_flank_score_v1;
        v1.bottom_flank_score = bottom_flank_score_v1;
        v1.flank_score = v1.use_top ? top_flank_score_v1 : bottom_flank_score_v1;
        v1.top_barcode_pos = {top_result_v1.startLocations[0], top_result_v1.endLocations[0]};
        v1.bottom_barcode_pos = {bottom_start + bottom_result_v1.startLocations[0],
                                 bottom_start + bottom_result_v1.endLocations[0]};

        // Calculate barcode scores for v2.
        auto top_mask_result_score_v2 =
                extract_mask_score(barcode2, top_mask_v2, mask_config, "top window v2");

        auto bottom_mask_result_score_v2 =
                extract_mask_score(barcode1_rev, bottom_mask_v2, mask_config, "bottom window v2");

        BarcodeScoreResult v2;
        v2.top_score = top_mask_result_score_v2;
        v2.bottom_score = bottom_mask_result_score_v2;
        v2.score = std::min(v2.top_score, v2.bottom_score);
        v2.use_top = v2.top_score < v2.bottom_score;
        v2.top_flank_score = top_flank_score_v2;
        v2.bottom_flank_score = bottom_flank_score_v2;
        v2.flank_score = v2.use_top ? top_flank_score_v2 : bottom_flank_score_v2;
        v2.top_barcode_pos = {top_result_v2.startLocations[0], top_result_v2.endLocations[0]};
        v2.bottom_barcode_pos = {bottom_start + bottom_result_v2.startLocations[0],
                                 bottom_start + bottom_result_v2.endLocations[0]};

        // The best score is the higher score between the 2 variants.
        const bool var1_is_best = v1.score < v2.score;
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
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);
    int bottom_start = std::max(0, (int)read_seq.length() - TRIM_LENGTH);
    std::string_view read_bottom = read_seq.substr(bottom_start, TRIM_LENGTH);

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
    auto top_start_idx = std::max(0, top_bc_loc - m_left_buffer - barcode_len);
    auto top_end_idx = top_bc_loc + m_right_buffer;
    std::string_view top_mask = read_top.substr(top_start_idx, top_end_idx - top_start_idx);

    auto [bottom_result, bottom_flank_score, bottom_bc_loc] = extract_flank_fit(
            bottom_context, read_bottom, barcode_len, placement_config, "bottom score");
    auto bottom_start_idx = std::max(0, bottom_bc_loc - m_left_buffer - barcode_len);
    auto bottom_end_idx = bottom_bc_loc + m_right_buffer;
    std::string_view bottom_mask =
            read_bottom.substr(bottom_start_idx, bottom_end_idx - bottom_start_idx);

    std::vector<BarcodeScoreResult> results;
    for (size_t i = 0; i < candidate.barcodes1.size(); i++) {
        auto barcode = top_left_buffer + candidate.barcodes1[i] + top_right_buffer;
        auto barcode_rev = bottom_left_buffer + candidate.barcodes1_rev[i] + bottom_right_buffer;
        auto& barcode_name = candidate.barcode_names[i];

        if (!barcode_is_permitted(allowed_barcodes, barcode_name)) {
            continue;
        }
        spdlog::trace("Checking barcode {}", barcode_name);

        auto top_mask_score = extract_mask_score(barcode, top_mask, mask_config, "top window");

        auto bottom_mask_score =
                extract_mask_score(barcode_rev, bottom_mask, mask_config, "bottom window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = candidate.kit;
        res.barcode_kit = candidate.barcode_kit;
        res.top_score = top_mask_score;
        res.bottom_score = bottom_mask_score;
        res.score = std::min(res.top_score, res.bottom_score);
        res.use_top = res.top_score < res.bottom_score;
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
        const BarcodeCandidateKit& candidate,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    std::string_view read_top = read_seq.substr(0, TRIM_LENGTH);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = init_edlib_config_for_flanks();

    EdlibAlignConfig mask_config = init_edlib_config_for_mask();

    std::string_view top_context = candidate.top_context;
    int barcode_len = int(candidate.barcodes1[0].length());

    auto [top_result, top_flank_score, top_bc_loc] =
            extract_flank_fit(top_context, read_top, barcode_len, placement_config, "top score");
    auto start_idx = std::max(0, top_bc_loc - m_left_buffer - barcode_len);
    auto end_idx = top_bc_loc + m_right_buffer;
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

        auto top_mask_score = extract_mask_score(barcode, top_mask, mask_config, "top window");

        BarcodeScoreResult res;
        res.barcode_name = barcode_name;
        res.kit = candidate.kit;
        res.barcode_kit = candidate.barcode_kit;
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
        const std::vector<BarcodeCandidateKit>& candidates,
        bool barcode_both_ends,
        const BarcodingInfo::FilterSet& allowed_barcodes) const {
    const std::string_view fwd = read_seq;

    // First find best barcode kit.
    const BarcodeCandidateKit* candidate;
    if (candidates.size() == 1) {
        candidate = &candidates[0];
    } else {
        // TODO: Implement finding best kit match.
        throw std::runtime_error("Unimplemented: multiple barcoding kits");
    }

    // Then find the best barcode hit within that kit.
    std::vector<BarcodeScoreResult> scores;
    const auto& kit = get_kit_info(candidate->kit);
    if (kit.double_ends) {
        if (kit.ends_different) {
            auto out = calculate_barcode_score_different_double_ends(fwd, *candidate,
                                                                     allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        } else {
            auto out = calculate_barcode_score_double_ends(fwd, *candidate, allowed_barcodes);
            scores.insert(scores.end(), out.begin(), out.end());
        }
    } else {
        auto out = calculate_barcode_score(fwd, *candidate, allowed_barcodes);
        scores.insert(scores.end(), out.begin(), out.end());
    }

    if (scores.empty()) {
        spdlog::warn("Barcode unclassified because no barcodes found in kit.");
        return UNCLASSIFIED;
    }

    if (kit.double_ends) {
        // For a double ended barcode, ensure that the best barcode according
        // to the top window and the best barcode according to the bottom window
        // are the same. If they suggest different barcodes confidently, then
        // consider the read unclassified.
        auto best_top_score = std::min_element(
                scores.begin(), scores.end(),
                [](const auto& l, const auto& r) { return l.top_score < r.top_score; });
        auto best_bottom_score = std::min_element(
                scores.begin(), scores.end(),
                [](const auto& l, const auto& r) { return l.bottom_score < r.bottom_score; });
        auto best_score = std::max(best_top_score->score, best_bottom_score->score);
        auto score_dist = std::abs(best_top_score->score - best_bottom_score->score);
        if ((best_score <= m_scoring_params.max_barcode_score) &&
            (score_dist <= m_scoring_params.min_barcode_score_dist) &&
            (best_top_score->barcode_name != best_bottom_score->barcode_name)) {
            spdlog::trace("Two ends confidently predict different BCs: top bc {}, bottom bc {}",
                          best_top_score->barcode_name, best_bottom_score->barcode_name);
            return UNCLASSIFIED;
        }
    }

    // Sort the scores windows by their barcode score.
    std::sort(scores.begin(), scores.end(),
              [](const auto& l, const auto& r) { return l.score < r.score; });

    std::stringstream d;
    for (auto& s : scores) {
        d << s.barcode_name << " " << s.score << ", ";
    }
    spdlog::trace("Scores: {}", d.str());
    auto best_score = scores.begin();
    auto are_scores_acceptable = [this](const auto& score) {
        return score.score <= m_scoring_params.max_barcode_score;
    };

    BarcodeScoreResult out = UNCLASSIFIED;
    if (scores.size() == 1) {
        if (are_scores_acceptable(*best_score)) {
            out = *best_score;
        }
    } else {
        const auto& second_best_score = std::next(best_score);
        const int score_dist = second_best_score->score - best_score->score;
        if (((score_dist >= m_scoring_params.min_barcode_score_dist &&
              are_scores_acceptable(*best_score)) ||
             (score_dist >= m_scoring_params.min_separation_only_dist)) &&
            (best_score->top_barcode_pos.first <= m_scoring_params.barcode_end_proximity ||
             best_score->bottom_barcode_pos.second >=
                     int(read_seq.length() - m_scoring_params.barcode_end_proximity))) {
            out = *best_score;
        }
    }

    if (barcode_both_ends && kit.double_ends) {
        // For more stringent classification, ensure that both ends of a read
        // have a high score for the same barcode. If not then consider it
        // unclassified.
        if (std::max(out.top_score, out.bottom_score) > m_scoring_params.max_barcode_score) {
            spdlog::trace("Min of top {} and bottom scores {} > max barcode score {}",
                          out.top_score, out.bottom_score, m_scoring_params.max_barcode_score);
            return UNCLASSIFIED;
        }
    }

    // If nothing is found, report as unclassified.
    return out;
}

}  // namespace demux

}  // namespace dorado
