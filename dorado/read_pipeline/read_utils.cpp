#include "read_utils.h"

#include "torch_utils/trim.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"

#include <ATen/TensorIndexing.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

using Slice = at::indexing::Slice;

namespace dorado::utils {
SimplexReadPtr shallow_copy_read(const SimplexRead& read) {
    auto copy = std::make_unique<SimplexRead>();
    copy->read_common.raw_data = read.read_common.raw_data;
    copy->digitisation = read.digitisation;
    copy->range = read.range;
    copy->offset = read.offset;
    copy->read_common.sample_rate = read.read_common.sample_rate;

    copy->read_common.scaling_method = read.read_common.scaling_method;
    copy->read_common.shift = read.read_common.shift;
    copy->read_common.scale = read.read_common.scale;

    copy->scaling = read.scaling;

    copy->read_common.model_stride = read.read_common.model_stride;

    copy->read_common.read_id = read.read_common.read_id;
    copy->read_common.seq = read.read_common.seq;
    copy->read_common.qstring = read.read_common.qstring;
    copy->read_common.moves = read.read_common.moves;
    copy->read_common.run_id = read.read_common.run_id;
    copy->read_common.flowcell_id = read.read_common.flowcell_id;
    copy->read_common.position_id = read.read_common.position_id;
    copy->read_common.experiment_id = read.read_common.experiment_id;
    copy->read_common.model_name = read.read_common.model_name;
    copy->read_common.sequencing_kit = read.read_common.sequencing_kit;

    copy->read_common.base_mod_probs = read.read_common.base_mod_probs;
    copy->read_common.mod_base_info = read.read_common.mod_base_info;

    copy->read_common.num_trimmed_samples = read.read_common.num_trimmed_samples;

    copy->read_common.is_rna_model = read.read_common.is_rna_model;
    copy->read_common.attributes = read.read_common.attributes;

    copy->start_sample = read.start_sample;
    copy->end_sample = read.end_sample;
    copy->run_acquisition_start_time_ms = read.run_acquisition_start_time_ms;
    copy->read_common.is_duplex = read.read_common.is_duplex;

    copy->read_common.read_tag = read.read_common.read_tag;
    copy->read_common.client_info = read.read_common.client_info;

    return copy;
}

int64_t find_mux_change_trim_seq_index(const std::string& qstring) {
    const int64_t size = static_cast<int64_t>(qstring.size());
    // This algorithm categorises qscores into low, mid and high. For each base in reverse, the
    // category score is accumulated and the index of the minimum value is taken as the
    // trim index (e.g. argmin).

    // Thresholds low:[0, 7], mid:(7, 12], high:(12, 50]
    // Add + 33 to avoid subtracting 33 from qsting
    const int kLowThreshold = 7 + 33;
    const int kHighThreshold = 12 + 33;
    // Scores [-1, 1, 10]
    const int kLowScore = -1;  // Do not change without updating early exit conditional
    const int kMidScore = 1;
    const int kHighScore = 10;

    int64_t trim_index = size - 1;  // index of minimum cumulative sum
    int cum_sum = 0;                // running total of cumulative sum
    int cum_sum_min = -1;           // minimum observed value

    for (int64_t i = size - 1; i >= 0; --i) {
        // Cast the qstring char to qscore. -33 is skipped by adding 33 to thresholds
        const int qs = static_cast<int>(qstring[i]);

        if (qs <= kLowThreshold) {
            cum_sum += kLowScore;
        } else if (qs <= kHighThreshold) {
            cum_sum += kMidScore;
        } else {
            cum_sum += kHighScore;
        }

        if (cum_sum <= cum_sum_min) {
            cum_sum_min = cum_sum;
            trim_index = i - 1;
        }

        // Early exit if cum_sum can't change by enough to change the result
        // This assumes kLowScore == -1
        if (cum_sum > i) {
            break;
        }
    }
    return trim_index;
}

void mux_change_trim_read(ReadCommon& read_common) {
    if (!read_common.attributes.is_end_reason_mux_change) {
        return;
    }

    const auto sequence_size = static_cast<int64_t>(read_common.qstring.size());

    // Do nothing for zero or very short sequences
    if (sequence_size < 100) {
        return;
    }

    const int64_t trim_seq_idx = find_mux_change_trim_seq_index(read_common.qstring);

    // Excessive trimming - do nothing
    if (trim_seq_idx < std::floor(sequence_size * 0.3f)) {
        spdlog::trace("mux_change_trimming {} - size: {} trim: {} excessive trimming",
                      read_common.read_id, sequence_size, trim_seq_idx);
        return;
    }

    const int kMinMuxChangeTrim = 5;
    // Nothing to do
    if (trim_seq_idx >= sequence_size - kMinMuxChangeTrim) {
        spdlog::trace("mux_change_trimming {} - no trim", read_common.read_id, trim_seq_idx);
        return;
    }

    // Trim the move table - We only trim from the back so no need to count leading trimmed samples
    const int64_t trim_moves_idx =
            utils::sequence_to_move_table_index(read_common.moves, trim_seq_idx, sequence_size);

    if (trim_moves_idx < 0) {
        spdlog::trace("mux_change_trimming {} - move table index failed", read_common.read_id);
        return;
    }
    read_common.moves.resize(trim_moves_idx);

    // Trim the sequence and qstring
    const std::pair<int, int> trim_interval = {0, int(trim_seq_idx)};
    read_common.seq = utils::trim_sequence(read_common.seq, trim_interval);
    read_common.qstring = utils::trim_sequence(read_common.qstring, trim_interval);

    // Trim the signal
    const size_t trim_signal_idx = read_common.moves.size() * read_common.model_stride;
    read_common.raw_data = read_common.raw_data.index({Slice(0, trim_signal_idx)});
    read_common.attributes.num_samples = read_common.get_raw_data_samples();

    spdlog::trace("mux_change_trimming {} - seq(before:{} after:{} net:-{})", read_common.read_id,
                  sequence_size, trim_seq_idx + 1, sequence_size - trim_seq_idx - 1);
}

}  // namespace dorado::utils
