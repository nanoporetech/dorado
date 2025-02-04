#include "Trimmer.h"

#include "read_pipeline/messages.h"
#include "torch_utils/trim.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <ATen/TensorIndexing.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

using Slice = at::indexing::Slice;

namespace {

// This part of trimming is split out into its own unoptimised function since not doing so
// causes binaries built by GCC8 with ASAN enabled to crash during static init.
// Note that the cause of the crash doesn't appear to be specific to this bit of code, since
// removing other parts from subread() also "fixes" the issue, but this is the smallest
// snippet that works around the issue without potentially incurring performance issues.
// This fix is copied from others parts of the code that exhibited similar errors
// with ASAN.
#if defined(__GNUC__) && defined(__SANITIZE_ADDRESS__)
__attribute__((optimize("O0")))
#endif
void trim_torch_tensor(at::Tensor& raw_data, std::pair<uint64_t,uint64_t> sample_trim_interval) {
    // Note - Duplex signal/read trimming is not supported yet.
    if (raw_data.sizes().size() > 1) {
        throw std::runtime_error("Read trimming is not supported for duplex reads");
    }
    raw_data = raw_data.index({Slice(sample_trim_interval.first, sample_trim_interval.second)});
}

}  // namespace

namespace dorado {

std::pair<int, int> Trimmer::determine_trim_interval(const BarcodeScoreResult& res, int seqlen) {
    // Initialize interval to be the whole read. Note that the interval
    // defines which portion of the read to retain.
    std::pair<int, int> trim_interval = {0, seqlen};

    if (res.kit == UNCLASSIFIED) {
        return trim_interval;
    }

    const float kFlankScoreThres = 0.6f;

    // Use barcode flank positions to determine trim interval
    // only if the flanks were confidently found. 1 is added to
    // the end of top barcode end value because that's the position
    // in the sequence where the barcode ends. So the actual sequence
    // because from one after that.
    if (res.top_penalty >= 0) {
        float top_flank_score = res.top_flank_score;
        if (top_flank_score > kFlankScoreThres) {
            trim_interval.first = res.top_barcode_pos.second + 1;
        }
    }

    if (res.bottom_penalty >= 0) {
        float bottom_flank_score = res.bottom_flank_score;
        if (bottom_flank_score > kFlankScoreThres) {
            trim_interval.second = res.bottom_barcode_pos.first;
        }
    }

    // In some cases where the read length is very small, the front
    // and rear windows could actually overlap. In that case find
    // which window was used and just grab the interval for that
    // window.
    if (trim_interval.second <= trim_interval.first) {
        if (res.use_top) {
            trim_interval = {res.top_barcode_pos.second + 1, seqlen};
        } else {
            trim_interval = {0, res.bottom_barcode_pos.first};
        }
    }

    if (trim_interval.second <= trim_interval.first) {
        // This could happen if the read is very short and the barcoding
        // algorithm determines the barcode interval to be the entire read.
        // In that case, skip trimming.
        trim_interval = {0, seqlen};
    }

    return trim_interval;
}

std::pair<int, int> Trimmer::determine_trim_interval(AdapterScoreResult& res, int seqlen) {
    // Initialize interval to be the whole read. Note that the interval
    // defines which portion of the read to retain.
    std::pair<int, int> trim_interval = {0, seqlen};

    const float score_thres = 0.8f;

    if (res.front.name == UNCLASSIFIED || res.front.score < score_thres) {
        trim_interval.first = 0;
        res.front.name = UNCLASSIFIED;
    } else {
        trim_interval.first = res.front.position.second + 1;
        spdlog::trace("Detected front interval adapter/primer - {}", res.front.name);
    }
    if (res.rear.name == UNCLASSIFIED || res.rear.score < score_thres) {
        trim_interval.second = seqlen;
        res.rear.name = UNCLASSIFIED;
    } else {
        trim_interval.second = res.rear.position.first;
        spdlog::trace("Detected rear interval adapter/primer - {}", res.rear.name);
    }

    if (trim_interval.second <= trim_interval.first) {
        // This could happen if the read is very short and the barcoding
        // algorithm determines the barcode interval to be the entire read.
        // In that case, skip trimming.
        trim_interval = {0, seqlen};
        res.front.name = UNCLASSIFIED;
        res.rear.name = UNCLASSIFIED;
    }

    return trim_interval;
}

BamPtr Trimmer::trim_sequence(bam1_t* input_record, std::pair<int, int> trim_interval) {
    bool is_seq_reversed = input_record->core.flag & BAM_FREVERSE;

    // Fetch components that need to be trimmed.
    std::string seq = utils::extract_sequence(input_record);
    std::vector<uint8_t> qual = utils::extract_quality(input_record);
    auto [stride, move_vals] = utils::extract_move_table(input_record);
    int ts = bam_aux_get(input_record, "ts") ? int(bam_aux2i(bam_aux_get(input_record, "ts"))) : -1;
    int ns = bam_aux_get(input_record, "ns") ? int(bam_aux2i(bam_aux_get(input_record, "ns"))) : -1;
    auto [modbase_str, modbase_probs] = utils::extract_modbase_info(input_record);

    // Any barcode/primer/adapter detection was done against the fwd sequence, so ensure we trim in that orientation too
    if (is_seq_reversed) {
        seq = utils::reverse_complement(seq);
        std::reverse(std::begin(qual), std::end(qual));
    }

    // Actually trim components.
    auto trimmed_seq = utils::trim_sequence(seq, trim_interval);
    auto trimmed_qual = utils::trim_quality(qual, trim_interval);
    auto [positions_trimmed, trimmed_moves] = utils::trim_move_table(move_vals, trim_interval);

    if (move_vals.empty()) {
        ns = -1;
        ts = -1;
    } else {
        if (ts >= 0) {
            ts += positions_trimmed * stride;
        }
        if (ns >= 0) {
            // After sequence trimming, the number of samples corresponding to the sequence is the size of
            // the new move table * stride. However, the ns tag includes the number of samples trimmed from
            // the front of the read as well. If ts is negative, the tag is not present, so treat it as 0.
            // |---------------------- ns ------------------|
            // |----ts----|--------moves signal-------------|
            ns = int(trimmed_moves.size() * stride) + std::max(0, ts);
        }
    }

    auto [trimmed_modbase_str, trimmed_modbase_probs] =
            utils::trim_modbase_info(seq, modbase_str, modbase_probs, trim_interval);

    // Create a new bam record to hold the trimmed read.
    BamPtr output(utils::new_unmapped_record(input_record, trimmed_seq, trimmed_qual));
    bam1_t* out_record = output.get();

    // Insert the new tags and delete the old ones.
    if (!trimmed_moves.empty()) {
        bam_aux_del(out_record, bam_aux_get(out_record, "mv"));
        // Move table format is stride followed by moves.
        trimmed_moves.insert(trimmed_moves.begin(), uint8_t(stride));
        bam_aux_update_array(out_record, "mv", 'c', int(trimmed_moves.size()),
                             (uint8_t*)trimmed_moves.data());
    }

    if (!trimmed_modbase_str.empty()) {
        bam_aux_del(out_record, bam_aux_get(out_record, "MM"));
        bam_aux_append(out_record, "MM", 'Z', int(trimmed_modbase_str.length() + 1),
                       (uint8_t*)trimmed_modbase_str.c_str());
        bam_aux_del(out_record, bam_aux_get(out_record, "ML"));
        bam_aux_update_array(out_record, "ML", 'C', int(trimmed_modbase_probs.size()),
                             (uint8_t*)trimmed_modbase_probs.data());
        bam_aux_update_int(out_record, "MN", trimmed_seq.length());
    }

    if (ts >= 0) {
        bam_aux_update_int(out_record, "ts", ts);
    } else if (bam_aux_get(out_record, "ts")) {
        bam_aux_del(out_record, bam_aux_get(out_record, "ts"));
    }
    if (ns >= 0) {
        bam_aux_update_int(out_record, "ns", ns);
    } else if (bam_aux_get(out_record, "ns")) {
        bam_aux_del(out_record, bam_aux_get(out_record, "ns"));
    }

    return output;
}

void Trimmer::trim_sequence(SimplexRead& read, std::pair<int, int> trim_interval) {
    if (trim_interval.second - trim_interval.first == int(read.read_common.seq.length())) {
        return;
    }

    read.read_common.seq = utils::trim_sequence(read.read_common.seq, trim_interval);
    read.read_common.qstring = utils::trim_sequence(read.read_common.qstring, trim_interval);

    auto [leading_mv_positions_trimmed, trimmed_moves] =
            utils::trim_move_table(read.read_common.moves, trim_interval);

    // Number of samples trimmed is the number of move positions trimmed from the front
    // of the read times the stride.
    auto num_leading_samples_trimmed = read.read_common.model_stride * leading_mv_positions_trimmed;
    // This gets added to the number of samples previously trimmed, such as from signal scaling, etc.
    read.read_common.num_trimmed_samples += num_leading_samples_trimmed;
    // The move table can be trimmed from both ends, so determine the new signal length corresponding
    // to the trimmed sequence by looking at new move table size.
    auto num_samples_from_mv_table = trimmed_moves.size() * read.read_common.model_stride;
    // The final signal should only correspond to the moves from the trimmed move table, so
    // the corresponding signal needs to be extracted from the original signal.
    trim_torch_tensor(
            read.read_common.raw_data,
            {num_leading_samples_trimmed, num_leading_samples_trimmed + num_samples_from_mv_table});

    read.read_common.moves = std::move(trimmed_moves);

    if (read.read_common.mod_base_info) {
        int num_modbase_channels = int(read.read_common.mod_base_info->alphabet.size());
        // The modbase probs table consists of the probability per channel per base. So when
        // trimming, we just shift everything by skipped bases * number of channels.
        std::pair<int, int> modbase_interval = {trim_interval.first * num_modbase_channels,
                                                trim_interval.second * num_modbase_channels};
        read.read_common.base_mod_probs =
                utils::trim_quality(read.read_common.base_mod_probs, modbase_interval);
    }
}

void Trimmer::check_and_update_barcoding(SimplexRead& read) {
    // If barcoding has been done, we may need to make some adjustments.
    if (!read.read_common.barcoding_result) {
        return;
    }
    auto& barcode_result = *read.read_common.barcoding_result;
    if (barcode_result.barcode_name == UNCLASSIFIED) {
        return;
    }

    bool front_barcode_trimmed = (read.read_common.barcode_trim_interval.first > 0);
    bool rear_barcode_trimmed = (read.read_common.barcode_trim_interval.second > 0 &&
                                 read.read_common.barcode_trim_interval.second <
                                         int(read.read_common.pre_trim_seq_length));

    if (front_barcode_trimmed || rear_barcode_trimmed) {
        // If barcodes are being trimmed, then the barcode positions are relative to the original
        // untrimmed sequence. So we don't need to make any updates.
        return;
    }

    bool front_adapter_trimmed = (read.read_common.adapter_trim_interval.first > 0);

    if (!front_adapter_trimmed) {
        // If we haven't trimmed any adapter or primer from the front of the read, then we don't need
        // to update any barcode positions.
        return;
    }

    // An adapter or primer was found trimmed from the beginning of the read, and barcodes were found,
    // but are not being trimmed. This means that their position details will need to be updated so that
    // they refer to the position in the trimmed read.
    if (barcode_result.top_barcode_pos != std::pair<int, int>(-1, -1)) {
        // We have detected, but not trimmed, a front barcode.
        // Update the position to correspond to the trimmed sequence.
        barcode_result.top_barcode_pos.first -= read.read_common.adapter_trim_interval.first;
        barcode_result.top_barcode_pos.second -= read.read_common.adapter_trim_interval.first;
    }
    if (barcode_result.bottom_barcode_pos != std::pair<int, int>(-1, -1)) {
        // We have detected, but not trimmed, a rear barcode.
        // Update position to correspond to the trimmed sequence.
        barcode_result.bottom_barcode_pos.first -= read.read_common.adapter_trim_interval.first;
        barcode_result.bottom_barcode_pos.second -= read.read_common.adapter_trim_interval.first;
    }
}

}  // namespace dorado
