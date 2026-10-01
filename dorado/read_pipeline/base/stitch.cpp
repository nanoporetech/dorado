#include "read_pipeline/base/stitch.h"

#include "read_pipeline/base/messages/ReadCommon.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <span>

namespace dorado::utils {

void stitch_chunks(ReadCommon& read_common, std::span<const Chunk*> called_chunks, bool is_vcs) {
    if (called_chunks.empty()) {
        return;
    }

    assert(std::all_of(std::begin(called_chunks), std::end(called_chunks), [&](const auto& chunk) {
        return static_cast<int>(div_round_closest(
                       pad_to(chunk->raw_chunk_size, chunk->chunk_size_granularity),
                       chunk->moves.size())) == read_common.attributes.model_stride;
    }));

    int start_pos = 0;
    int mid_point_front = 0;
    std::vector<uint8_t> moves;
    std::vector<std::string_view> sequences;
    std::vector<std::string_view> qstrings;
    std::vector<uint8_t> base_mod_probs;
    std::vector<bool> base_mod_simplex_motif_hits;

    const auto mod_chunk = std::find_if(called_chunks.begin(), called_chunks.end(),
                                        [](auto chunk) { return !chunk->base_mod_probs.empty(); });
    const size_t num_states =
            mod_chunk == called_chunks.end()
                    ? 0
                    : (*mod_chunk)->base_mod_probs.size() / (*mod_chunk)->seq.size();

    auto append_modbase_data = [&](const Chunk& chunk, size_t start, size_t length) {
        if (num_states != 0) {
            assert(chunk.base_mod_probs.size() == chunk.seq.size() * num_states);
            const auto begin = chunk.base_mod_probs.begin() + start * num_states;
            base_mod_probs.insert(base_mod_probs.end(), begin, begin + length * num_states);
        }
        if (!chunk.base_mod_simplex_motif_hits.empty()) {
            assert(chunk.base_mod_simplex_motif_hits.size() == chunk.seq.size());
            const auto begin = chunk.base_mod_simplex_motif_hits.begin() + start;
            base_mod_simplex_motif_hits.insert(base_mod_simplex_motif_hits.end(), begin,
                                               begin + length);
        }
    };

    sequences.reserve(called_chunks.size());
    qstrings.reserve(called_chunks.size());

    for (int i = 0; i < int(called_chunks.size()) - 1; i++) {
        auto& current_chunk = called_chunks[i];
        auto& next_chunk = called_chunks[i + 1];
        const int overlap_size = int((current_chunk->raw_chunk_size + current_chunk->input_offset) -
                                     (next_chunk->input_offset));
        assert(overlap_size % read_common.attributes.model_stride == 0);
        const int overlap_down_sampled = overlap_size / read_common.attributes.model_stride;
        const int mid_point_rear = overlap_down_sampled / 2;

        const int current_chunk_bases_to_trim =
                std::reduce(std::prev(current_chunk->moves.end(), mid_point_rear),
                            current_chunk->moves.end(), 0);

        const int current_chunk_seq_len = int(current_chunk->seq.size());
        const int end_pos = current_chunk_seq_len - current_chunk_bases_to_trim;
        const int trimmed_len = end_pos - start_pos;
        const std::string_view seq = current_chunk->seq;
        const std::string_view qstring = current_chunk->qstring;
        sequences.push_back(seq.substr(start_pos, trimmed_len));
        qstrings.push_back(qstring.substr(start_pos, trimmed_len));
        append_modbase_data(*current_chunk, start_pos, trimmed_len);

        moves.insert(moves.end(), std::next(current_chunk->moves.begin(), mid_point_front),
                     std::prev(current_chunk->moves.end(), mid_point_rear));

        mid_point_front = overlap_down_sampled - mid_point_rear;

        start_pos = 0;
        for (int j = 0; j < mid_point_front; j++) {
            start_pos += (int)next_chunk->moves[j];
        }
    }

    // Append the final chunk
    auto& last_chunk = called_chunks.back();
    const std::string_view last_seq = last_chunk->seq;
    const std::string_view last_qstring = last_chunk->qstring;
    auto last_moves = std::span<const uint8_t>(last_chunk->moves);

    if (called_chunks.size() == 1 || is_vcs) {
        // shorten the sequence, qstring & moves where the actual read signal is shorter than chunksize
        const int signal_size =
                std::min(read_common.get_raw_data_samples(), last_chunk->raw_chunk_size);
        const int last_index_in_moves_to_keep =
                div_round_up(signal_size, read_common.attributes.model_stride);
        last_moves =
                last_moves.subspan(mid_point_front, last_index_in_moves_to_keep - mid_point_front);
        const int end = std::reduce(last_moves.begin(), last_moves.end(), 0);
        sequences.push_back(last_seq.substr(start_pos, end));
        qstrings.push_back(last_qstring.substr(start_pos, end));
        append_modbase_data(*last_chunk, start_pos, end);

    } else {
        sequences.push_back(last_seq.substr(start_pos));
        qstrings.push_back(last_qstring.substr(start_pos));
        append_modbase_data(*last_chunk, start_pos, last_seq.size() - start_pos);
        last_moves = last_moves.subspan(mid_point_front);
    }

    // Set the read seq and qstring
    read_common.seq = utils::join(sequences, {});
    read_common.qstring = utils::join(qstrings, {});
    read_common.base_mod_probs = std::move(base_mod_probs);
    read_common.base_mod_simplex_motif_hits = std::move(base_mod_simplex_motif_hits);
    read_common.moves = std::move(moves);
    read_common.moves.insert(std::end(read_common.moves), std::begin(last_moves),
                             std::end(last_moves));

    // remove partial stride overhang
    if (static_cast<int>(read_common.moves.size()) >
        static_cast<int>(read_common.get_raw_data_samples() /
                         read_common.attributes.model_stride)) {
        if (read_common.moves.back() == 1) {
            read_common.seq.pop_back();
            read_common.qstring.pop_back();
            if (num_states != 0) {
                read_common.base_mod_probs.resize(read_common.base_mod_probs.size() - num_states);
            }
            if (!read_common.base_mod_simplex_motif_hits.empty()) {
                read_common.base_mod_simplex_motif_hits.pop_back();
            }
        }
        read_common.moves.pop_back();
        assert(size_t(std::reduce(read_common.moves.begin(), read_common.moves.end(), 0)) ==
               read_common.seq.size());
    }

    assert(num_states == 0 ||
           read_common.base_mod_probs.size() == read_common.seq.size() * num_states);
    assert(read_common.base_mod_simplex_motif_hits.empty() ||
           read_common.base_mod_simplex_motif_hits.size() == read_common.seq.size());
}

}  // namespace dorado::utils
