#include "stitch.h"

#include "ReadPipeline.h"
#include "utils/math_utils.h"

#include <algorithm>
#include <cassert>

namespace dorado::utils {

void stitch_chunks(ReadCommon& read_common,
                   const std::vector<std::unique_ptr<Chunk>>& called_chunks) {
    assert(static_cast<int>(div_round_closest(called_chunks[0]->raw_chunk_size,
                                              called_chunks[0]->moves.size())) ==
           read_common.model_stride);

    int start_pos = 0;
    int mid_point_front = 0;
    std::vector<uint8_t> moves;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;

    for (int i = 0; i < int(called_chunks.size() - 1); i++) {
        auto& current_chunk = called_chunks[i];
        auto& next_chunk = called_chunks[i + 1];
        int overlap_size = int((current_chunk->raw_chunk_size + current_chunk->input_offset) -
                               (next_chunk->input_offset));
        assert(overlap_size % read_common.model_stride == 0);
        int overlap_down_sampled = overlap_size / read_common.model_stride;
        int mid_point_rear = overlap_down_sampled / 2;

        int current_chunk_bases_to_trim =
                std::accumulate(std::prev(current_chunk->moves.end(), mid_point_rear),
                                current_chunk->moves.end(), 0);

        int current_chunk_seq_len = int(current_chunk->seq.size());
        int end_pos = current_chunk_seq_len - current_chunk_bases_to_trim;
        int trimmed_len = end_pos - start_pos;
        sequences.push_back(current_chunk->seq.substr(start_pos, trimmed_len));
        qstrings.push_back(current_chunk->qstring.substr(start_pos, trimmed_len));
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
    moves.insert(moves.end(), std::next(last_chunk->moves.begin(), mid_point_front),
                 last_chunk->moves.end());

    if (called_chunks.size() == 1) {
        // shorten the sequence, qstring & moves where the read is shorter than chunksize
        int last_index_in_moves_to_keep =
                int(read_common.get_raw_data_samples() / read_common.model_stride);
        moves = std::vector<uint8_t>(moves.begin(), moves.begin() + last_index_in_moves_to_keep);
        int end = std::accumulate(moves.begin(), moves.end(), 0);
        sequences.push_back(last_chunk->seq.substr(start_pos, end));
        qstrings.push_back(last_chunk->qstring.substr(start_pos, end));

    } else {
        sequences.push_back(last_chunk->seq.substr(start_pos));
        qstrings.push_back(last_chunk->qstring.substr(start_pos));
    }

    // Set the read seq and qstring
    read_common.seq = std::accumulate(sequences.begin(), sequences.end(), std::string(""));
    read_common.qstring = std::accumulate(qstrings.begin(), qstrings.end(), std::string(""));
    read_common.moves = std::move(moves);

    // remove partial stride overhang
    if (static_cast<int>(read_common.moves.size()) >
        static_cast<int>(read_common.get_raw_data_samples() / read_common.model_stride)) {
        if (read_common.moves.back() == 1) {
            read_common.seq.pop_back();
            read_common.qstring.pop_back();
        }
        read_common.moves.pop_back();
        assert(size_t(std::accumulate(read_common.moves.begin(), read_common.moves.end(), 0)) ==
               read_common.seq.size());
    }
}

}  // namespace dorado::utils
