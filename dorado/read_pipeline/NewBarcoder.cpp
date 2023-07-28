#include "NewBarcoder.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "htslib/sam.h"
#include "utils/alignment_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace {

// Calculate the edit distance for an alignment just within the region
// which maps to the barcode sequence. i.e. Ignore any edits made to the
// flanking regions.
int calculate_edit_dist(const EdlibAlignResult& res, int flank_len, int query_len) {
    int dist = 0;
    int qpos = 0;
    for (int i = 0; i < res.alignmentLength; i++) {
        if (qpos < flank_len) {
            if (res.alignment[i] == EDLIB_EDOP_MATCH) {
                qpos++;
            } else if (res.alignment[i] == EDLIB_EDOP_MISMATCH) {
                qpos++;
            } else if (res.alignment[i] == EDLIB_EDOP_DELETE) {
            } else if (res.alignment[i] == EDLIB_EDOP_INSERT) {
                qpos++;
            }
            //std::cerr << qpos << ", " << i << std::endl;
        } else {
            if (query_len == 0) {
                break;
            }
            if (res.alignment[i] == EDLIB_EDOP_MATCH) {
                query_len--;
            } else if (res.alignment[i] == EDLIB_EDOP_MISMATCH) {
                dist++;
                query_len--;
            } else if (res.alignment[i] == EDLIB_EDOP_DELETE) {
                dist += 1;
            } else if (res.alignment[i] == EDLIB_EDOP_INSERT) {
                query_len--;
                dist += 1;
            }
        }
    }
    return dist;
}

}  // namespace

const std::string UNCLASSIFIED_BARCODE = "unclassified";

BarcoderNode::BarcoderNode(int threads, const std::vector<std::string>& kit_names)
        : MessageSink(10000), m_threads(threads), m_barcoder(kit_names) {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&BarcoderNode::worker_thread, this, i)));
    }
}

void BarcoderNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
}

BarcoderNode::~BarcoderNode() {
    terminate_impl();
    spdlog::info("> Barcoded: {}", m_matched.load());
    spdlog::info("> Bases Processed: {}", m_bases.load());
}

void BarcoderNode::worker_thread(size_t tid) {
    Message message;
    while (m_work_queue.try_pop(message)) {
        auto read = std::get<BamPtr>(std::move(message));
        auto records = barcode(read.get());
        for (auto& record : records) {
            send_message_to_sink(std::move(record));
        }
    }
}

std::vector<BamPtr> BarcoderNode::barcode(bam1_t* irecord) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;
    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    m_bases += seq.length();

    auto bc_res = m_barcoder.barcode(seq);
    auto bc = (bc_res.adapter_name == UNCLASSIFIED_BARCODE)
                      ? UNCLASSIFIED_BARCODE
                      : bc_res.kit + "_" + bc_res.adapter_name;
    bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
    if (bc != UNCLASSIFIED_BARCODE) {
        m_matched++;
    }
    results.push_back(BamPtr(bam_dup1(irecord)));

    return results;
}

stats::NamedStats BarcoderNode::sample_stats() const { return stats::from_obj(m_work_queue); }

Barcoder::Barcoder(const std::vector<std::string>& kit_names) {
    m_adapter_sequences = generate_adapter_sequence(kit_names);
}

ScoreResults Barcoder::barcode(const std::string& seq) {
    auto best_adapter = find_best_adapter(seq, m_adapter_sequences);
    return best_adapter;
}

// Generate all possible barcode adapters. If kit name is passed
// limit the adapters generated to only the specified kits.
// Returns a vector all barcode adapter sequences to test the
// input read sequence against.
std::vector<AdapterSequence> Barcoder::generate_adapter_sequence(
        const std::vector<std::string>& kit_names) {
    std::vector<AdapterSequence> adapters;
    std::vector<std::string> final_kit_names;
    if (kit_names.size() == 0) {
        for (auto& [kit_name, kit] : kit_info) {
            final_kit_names.push_back(kit_name);
        }
    } else {
        final_kit_names = kit_names;
    }
    spdlog::debug("> Kits to evaluate: {}", final_kit_names.size());

    for (auto& kit_name : final_kit_names) {
        auto kit_info = dorado::kit_info.at(kit_name);
        AdapterSequence as;
        as.kit = kit_name;
        auto& ref_bc = barcodes.at(kit_info.barcodes[0]);

        as.top_primer = kit_info.top_front_flank + std::string(ref_bc.length(), 'N') +
                        kit_info.top_rear_flank;
        as.top_primer_rev = utils::reverse_complement(kit_info.top_rear_flank) +
                            std::string(ref_bc.length(), 'N') +
                            utils::reverse_complement(kit_info.top_front_flank);
        as.bottom_primer = kit_info.bottom_front_flank + std::string(ref_bc.length(), 'N') +
                           kit_info.bottom_rear_flank;
        as.bottom_primer_rev = utils::reverse_complement(kit_info.bottom_rear_flank) +
                               std::string(ref_bc.length(), 'N') +
                               utils::reverse_complement(kit_info.bottom_front_flank);

        for (auto& bc_name : kit_info.barcodes) {
            auto adapter = barcodes.at(bc_name);
            auto adapter_rev = utils::reverse_complement(adapter);

            as.adapter.push_back(adapter);
            as.adapter_rev.push_back(adapter_rev);

            as.adapter_name.push_back(bc_name);
        }
        adapters.push_back(as);
    }
    return adapters;
}
int extract_mask_location(EdlibAlignResult aln, const std::string_view& query) {
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

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====----------------=====-> 3'
//      BCXX_1             RC(BCXX_2)
//
// 3' <-=====----------------=====-< 5'
//    RC(BCXX_1)             BCXX_2
//
// In this scenario, the barcode (and its flanks) ligate to both ends
// of the read. The adapter sequence is also different for top and bottom strands.
// So we need to check bottom ends of the read. Since the adapters always ligate to
// 5' end of the read, the 3' end of the other strand has the reverse complement
// of that adapter sequence.
void Barcoder::calculate_adapter_score_different_double_ends(const std::string_view& read_seq,
                                                             const AdapterSequence& as,
                                                             std::vector<ScoreResults>& results) {
    if (read_seq.length() < 150) {
        return;
    }
    std::string_view read_top = read_seq.substr(0, 150);
    std::string_view read_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_PATH;
    EdlibEqualityPair additionalEqualities[4] = {{'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 4;

    EdlibAlignConfig mask_config = edlibDefaultAlignConfig();
    mask_config.mode = EDLIB_MODE_NW;
    mask_config.task =
            (spdlog::get_level() == spdlog::level::debug) ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;

    // Fetch barcode mask locations for variant 1
    std::string_view top_strand_v1 = as.top_primer;
    std::string_view bottom_strand_v1 = as.bottom_primer_rev;
    std::string_view top_strand_v2 = as.bottom_primer;
    std::string_view bottom_strand_v2 = as.top_primer_rev;

    EdlibAlignResult top_result_v1 =
            edlibAlign(top_strand_v1.data(), top_strand_v1.length(), read_top.data(),
                       read_top.length(), placement_config);
    spdlog::debug("top score v1 {}", top_result_v1.editDistance);
    spdlog::debug("\n{}",
                  utils::alignment_to_str(top_strand_v1.data(), read_top.data(), top_result_v1));
    int top_bc_loc_v1 = extract_mask_location(top_result_v1, top_strand_v1);
    const std::string_view& top_mask_v1 = read_top.substr(top_bc_loc_v1, as.adapter[0].length());
    float top_flank_score_v1 = 1.f - static_cast<float>(top_result_v1.editDistance) /
                                             (top_strand_v1.length() - as.adapter[0].length());

    EdlibAlignResult bottom_result_v1 =
            edlibAlign(bottom_strand_v1.data(), bottom_strand_v1.length(), read_bottom.data(),
                       read_bottom.length(), placement_config);
    spdlog::debug("bottom score v1 {}", bottom_result_v1.editDistance);
    spdlog::debug("\n{}", utils::alignment_to_str(bottom_strand_v1.data(), read_bottom.data(),
                                                  bottom_result_v1));
    int bottom_bc_loc_v1 = extract_mask_location(bottom_result_v1, bottom_strand_v1);
    const std::string_view& bottom_mask_v1 =
            read_bottom.substr(bottom_bc_loc_v1, as.adapter_rev[0].length());
    float bottom_flank_score_v1 =
            1.f - static_cast<float>(bottom_result_v1.editDistance) /
                          (bottom_strand_v1.length() - as.adapter[0].length());

    // Fetch barcode mask locations for variant 2
    EdlibAlignResult top_result_v2 =
            edlibAlign(top_strand_v2.data(), top_strand_v2.length(), read_top.data(),
                       read_top.length(), placement_config);
    spdlog::debug("top score v2 {}", top_result_v2.editDistance);
    spdlog::debug("\n{}",
                  utils::alignment_to_str(top_strand_v2.data(), read_top.data(), top_result_v2));
    int top_bc_loc_v2 = extract_mask_location(top_result_v2, top_strand_v2);
    const std::string_view& top_mask_v2 = read_top.substr(top_bc_loc_v2, as.adapter[0].length());
    float top_flank_score_v2 = 1.f - static_cast<float>(top_result_v2.editDistance) /
                                             (top_strand_v2.length() - as.adapter[0].length());

    EdlibAlignResult bottom_result_v2 =
            edlibAlign(bottom_strand_v2.data(), bottom_strand_v2.length(), read_bottom.data(),
                       read_bottom.length(), placement_config);
    spdlog::debug("bottom score v2 {}", bottom_result_v2.editDistance);
    spdlog::debug("\n{}", utils::alignment_to_str(bottom_strand_v2.data(), read_bottom.data(),
                                                  bottom_result_v2));
    int bottom_bc_loc_v2 = extract_mask_location(bottom_result_v2, bottom_strand_v2);
    const std::string_view& bottom_mask_v2 =
            read_bottom.substr(bottom_bc_loc_v2, as.adapter_rev[0].length());
    float bottom_flank_score_v2 =
            1.f - static_cast<float>(bottom_result_v2.editDistance) /
                          (bottom_strand_v2.length() - as.adapter[0].length());

    // Find the best variant of the two.
    int total_v1_score = top_result_v1.editDistance + bottom_result_v1.editDistance;
    int total_v2_score = top_result_v2.editDistance + bottom_result_v2.editDistance;

    spdlog::debug("best variant {}", (total_v1_score < total_v2_score) ? "v1" : "v2");
    const auto& top_mask = (total_v1_score < total_v2_score) ? top_mask_v1 : top_mask_v2;
    const auto& bottom_mask = (total_v1_score < total_v2_score) ? bottom_mask_v1 : bottom_mask_v2;
    float top_flank_score =
            (total_v1_score < total_v2_score) ? top_flank_score_v1 : top_flank_score_v2;
    float bottom_flank_score =
            (total_v1_score < total_v2_score) ? bottom_flank_score_v1 : bottom_flank_score_v2;

    //std::vector<ScoreResults> results;
    for (int i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_rev = as.adapter_rev[i];
        auto& adapter_name = as.adapter_name[i];
        spdlog::debug("Barcoder {}", adapter_name);

        // Calculate barcode scores for v1.
        auto top_mask_result_v1 = edlibAlign(adapter.data(), adapter.length(), top_mask_v1.data(),
                                             top_mask_v1.length(), mask_config);
        float top_mask_result_score_v1 =
                1.f - static_cast<float>(top_mask_result_v1.editDistance) / adapter.length();
        spdlog::debug("top window v1 {}", top_mask_result_v1.editDistance);
        spdlog::debug("\n{}", utils::alignment_to_str(adapter.data(), top_mask_v1.data(),
                                                      top_mask_result_v1));

        auto bottom_mask_result_v1 =
                edlibAlign(adapter_rev.data(), adapter_rev.length(), bottom_mask_v1.data(),
                           bottom_mask_v1.length(), mask_config);
        float bottom_mask_result_score_v1 =
                1.f - static_cast<float>(bottom_mask_result_v1.editDistance) / adapter_rev.length();

        spdlog::debug("bottom window v1 {}", bottom_mask_result_v1.editDistance);
        spdlog::debug("\n{}", utils::alignment_to_str(adapter_rev.data(), bottom_mask_v1.data(),
                                                      bottom_mask_result_v1));

        ScoreResults v1;
        v1.top_score = top_mask_result_score_v1;
        v1.bottom_score = bottom_mask_result_score_v1;
        v1.score = std::max(v1.top_score, v1.bottom_score);
        v1.top_flank_score = top_flank_score_v1;
        v1.bottom_flank_score = bottom_flank_score_v1;
        v1.flank_score =
                (v1.top_score > v1.bottom_score) ? top_flank_score_v1 : bottom_flank_score_v1;

        // Calculate barcode scores for v2.
        auto top_mask_result_v2 = edlibAlign(adapter.data(), adapter.length(), top_mask_v2.data(),
                                             top_mask_v2.length(), mask_config);
        float top_mask_result_score_v2 =
                1.f - static_cast<float>(top_mask_result_v2.editDistance) / adapter.length();
        spdlog::debug("top window v2 {}", top_mask_result_v2.editDistance);
        spdlog::debug("\n{}", utils::alignment_to_str(adapter.data(), top_mask_v2.data(),
                                                      top_mask_result_v2));

        auto bottom_mask_result_v2 =
                edlibAlign(adapter_rev.data(), adapter_rev.length(), bottom_mask_v2.data(),
                           bottom_mask_v2.length(), mask_config);
        float bottom_mask_result_score_v2 =
                1.f - static_cast<float>(bottom_mask_result_v2.editDistance) / adapter_rev.length();

        spdlog::debug("bottom window v2 {}", bottom_mask_result_v2.editDistance);
        spdlog::debug("\n{}", utils::alignment_to_str(adapter_rev.data(), bottom_mask_v2.data(),
                                                      bottom_mask_result_v2));

        ScoreResults v2;
        v2.top_score = top_mask_result_score_v2;
        v2.bottom_score = bottom_mask_result_score_v2;
        v2.score = std::max(v2.top_score, v2.bottom_score);
        v2.top_flank_score = top_flank_score_v2;
        v2.bottom_flank_score = bottom_flank_score_v2;
        v2.flank_score =
                (v2.top_score > v2.bottom_score) ? top_flank_score_v2 : bottom_flank_score_v2;

        ScoreResults res = (v1.score > v2.score) ? v1 : v2;
        res.adapter_name = adapter_name;
        res.kit = as.kit;

        edlibFreeAlignResult(top_mask_result_v1);
        edlibFreeAlignResult(bottom_mask_result_v1);
        edlibFreeAlignResult(top_mask_result_v2);
        edlibFreeAlignResult(bottom_mask_result_v2);

        results.push_back(res);
    }
    edlibFreeAlignResult(top_result_v1);
    edlibFreeAlignResult(bottom_result_v1);
    edlibFreeAlignResult(top_result_v2);
    edlibFreeAlignResult(bottom_result_v2);
    return;
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
void Barcoder::calculate_adapter_score_double_ends(const std::string_view& read_seq,
                                                   const AdapterSequence& as,
                                                   std::vector<ScoreResults>& results) {
    if (read_seq.length() < 150) {
        return;
    }
    bool debug_mode = (spdlog::get_level() == spdlog::level::debug);
    std::string_view read_top = read_seq.substr(0, 150);
    std::string_view read_bottom = read_seq.substr(std::max(0, (int)read_seq.length() - 150), 150);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_PATH;
    EdlibEqualityPair additionalEqualities[4] = {{'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 4;

    EdlibAlignConfig mask_config = edlibDefaultAlignConfig();
    mask_config.mode = EDLIB_MODE_NW;
    mask_config.task = debug_mode ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;

    std::string_view top_strand;
    std::string_view bottom_strand;
    top_strand = as.top_primer;
    bottom_strand = as.top_primer_rev;

    EdlibAlignResult top_result = edlibAlign(top_strand.data(), top_strand.length(),
                                             read_top.data(), read_top.length(), placement_config);
    if (debug_mode) {
        spdlog::debug("top score {}", top_result.editDistance);
        spdlog::debug("\n{}",
                      utils::alignment_to_str(top_strand.data(), read_top.data(), top_result));
    }
    int top_bc_loc = extract_mask_location(top_result, top_strand);
    const std::string_view& top_mask = read_top.substr(top_bc_loc, as.adapter[0].length());
    float top_flank_score = 1.f - static_cast<float>(top_result.editDistance) /
                                          (top_strand.length() - as.adapter[0].length());

    EdlibAlignResult bottom_result =
            edlibAlign(bottom_strand.data(), bottom_strand.length(), read_bottom.data(),
                       read_bottom.length(), placement_config);
    if (debug_mode) {
        spdlog::debug("bottom score {}", bottom_result.editDistance);
        spdlog::debug("\n{}", utils::alignment_to_str(bottom_strand.data(), read_bottom.data(),
                                                      bottom_result));
    }
    int bottom_bc_loc = extract_mask_location(bottom_result, bottom_strand);
    const std::string_view& bottom_mask =
            read_bottom.substr(bottom_bc_loc, as.adapter_rev[0].length());
    float bottom_flank_score = 1.f - static_cast<float>(bottom_result.editDistance) /
                                             (bottom_strand.length() - as.adapter_rev[0].length());

    //std::vector<ScoreResults> results;
    for (int i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_rev = as.adapter_rev[i];
        auto& adapter_name = as.adapter_name[i];
        spdlog::debug("Barcoder {}", adapter_name);

        auto top_mask_result = edlibAlign(adapter.data(), adapter.length(), top_mask.data(),
                                          top_mask.length(), mask_config);
        if (debug_mode) {
            spdlog::debug("top window {}", top_mask_result.editDistance);
            spdlog::debug("\n{}", utils::alignment_to_str(adapter.data(), top_mask.data(),
                                                          top_mask_result));
        }

        auto bottom_mask_result = edlibAlign(adapter_rev.data(), adapter_rev.length(),
                                             bottom_mask.data(), bottom_mask.length(), mask_config);

        if (debug_mode) {
            spdlog::debug("bottom window {}", bottom_mask_result.editDistance);
            spdlog::debug("\n{}", utils::alignment_to_str(adapter_rev.data(), bottom_mask.data(),
                                                          bottom_mask_result));
        }

        ScoreResults res;
        res.adapter_name = adapter_name;
        res.kit = as.kit;
        res.top_flank_score = top_flank_score;
        res.bottom_flank_score = bottom_flank_score;
        res.flank_score = std::max(res.top_flank_score, res.bottom_flank_score);
        res.top_score = 1.f - static_cast<float>(top_mask_result.editDistance) / adapter.length();
        res.bottom_score =
                1.f - static_cast<float>(bottom_mask_result.editDistance) / adapter_rev.length();
        res.score = std::max(res.top_score, res.bottom_score);

        edlibFreeAlignResult(top_mask_result);
        edlibFreeAlignResult(bottom_mask_result);
        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    edlibFreeAlignResult(bottom_result);
    return;
}

// Calculate barcode score for the following barcoding scenario:
// 5' >-=====---------------> 3'
//      BCXXX
//
// In this scenario, the barcode (and its flanks) only ligate to the 5' end
// of the read. So we only look for adapter sequence in the top "window" (first
// 150bp) of the read.
void Barcoder::calculate_adapter_score(const std::string_view& read_seq,
                                       const AdapterSequence& as,
                                       std::vector<ScoreResults>& results) {
    if (read_seq.length() < 150) {
        return;
    }
    bool debug_mode = (spdlog::get_level() == spdlog::level::debug);
    std::string_view read_top = read_seq.substr(0, 150);

    // Try to find the location of the barcode + flanks in the top and bottom windows.
    EdlibAlignConfig placement_config = edlibDefaultAlignConfig();
    placement_config.mode = EDLIB_MODE_HW;
    placement_config.task = EDLIB_TASK_PATH;
    EdlibEqualityPair additionalEqualities[4] = {{'N', 'A'}, {'N', 'T'}, {'N', 'C'}, {'N', 'G'}};
    placement_config.additionalEqualities = additionalEqualities;
    placement_config.additionalEqualitiesLength = 4;

    EdlibAlignConfig mask_config = edlibDefaultAlignConfig();
    mask_config.mode = EDLIB_MODE_NW;
    mask_config.task = debug_mode ? EDLIB_TASK_PATH : EDLIB_TASK_LOC;

    std::string_view top_strand;
    top_strand = as.top_primer;

    EdlibAlignResult top_result = edlibAlign(top_strand.data(), top_strand.length(),
                                             read_top.data(), read_top.length(), placement_config);
    if (debug_mode) {
        spdlog::debug("top score {}", top_result.editDistance);
        spdlog::debug("\n{}",
                      utils::alignment_to_str(top_strand.data(), read_top.data(), top_result));
    }
    int top_bc_loc = extract_mask_location(top_result, top_strand);
    const std::string_view& top_mask = read_top.substr(top_bc_loc, as.adapter[0].length());

    //std::vector<ScoreResults> results;
    for (int i = 0; i < as.adapter.size(); i++) {
        auto& adapter = as.adapter[i];
        auto& adapter_name = as.adapter_name[i];
        spdlog::debug("Barcoder {}", adapter_name);

        auto top_mask_result = edlibAlign(adapter.data(), adapter.length(), top_mask.data(),
                                          top_mask.length(), mask_config);
        if (debug_mode) {
            spdlog::debug("top window {}", top_mask_result.editDistance);
            spdlog::debug("\n{}", utils::alignment_to_str(adapter.data(), top_mask.data(),
                                                          top_mask_result));
        }

        ScoreResults res;
        res.adapter_name = adapter_name;
        res.kit = as.kit;
        res.top_flank_score = 1.f - static_cast<float>(top_result.editDistance) /
                                            (top_strand.length() - adapter.length());
        res.bottom_flank_score = -1.f;
        res.flank_score = std::max(res.top_flank_score, res.bottom_flank_score);
        res.top_score = 1.f - static_cast<float>(top_mask_result.editDistance) / adapter.length();
        res.bottom_score = -1.f;
        res.score = std::max(res.top_score, res.bottom_score);

        edlibFreeAlignResult(top_mask_result);
        results.push_back(res);
    }
    edlibFreeAlignResult(top_result);
    return;
}

// Score every barcode against the input read and returns the best match,
// or an unclassified match, based on certain heuristics.
ScoreResults Barcoder::find_best_adapter(const std::string& read_seq,
                                         std::vector<AdapterSequence>& adapters) {
    std::string fwd = read_seq;

    std::vector<ScoreResults> scores;
    for (auto& as : adapters) {
        auto& kit = kit_info.at(as.kit);
        if (kit.double_ends) {
            if (kit.ends_different) {
                calculate_adapter_score_different_double_ends(fwd, as, scores);
            } else {
                calculate_adapter_score_double_ends(fwd, as, scores);
            }
        } else {
            calculate_adapter_score(fwd, as, scores);
        }
    }

    // Sore the scores windows by their adapter score.
    std::sort(scores.begin(), scores.end(),
              [](const auto& l, const auto& r) { return l.score > r.score; });
    auto best_score = scores.begin();
    // At minimum, the best window must meet the adapter score threshold.
    //spdlog::debug("Best candidate from list {} flank {} barcode {}", best_score->score,
    //              best_score->flank_score, best_score->adapter_name);
    std::string d = "";
    for (auto& s : scores) {
        d += std::to_string(s.score) + " " + s.adapter_name + ", ";
    }
    spdlog::debug("Scores: {}", d);
    const float kThresBc = 0.7f;
    const float kThresFlank = 0.5f;
    const float kMargin = 0.05f;
    //if (best_score != scores.end() && best_score->score >= kThresBc) {
    if (best_score != scores.end()) {  // && best_score->score >= kThresBc) {
        if (best_score->flank_score >= 0.7 && best_score->score >= 0.66) {
            return *best_score;
        } else if (best_score->score >= 0.7 && best_score->flank_score >= 0.6) {
            return *best_score;
        }
    }

    // If nothing is found, report as unclassified.
    return {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, UNCLASSIFIED_BARCODE, UNCLASSIFIED_BARCODE};
}

}  // namespace dorado
