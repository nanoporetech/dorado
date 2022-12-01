#include "../read_pipeline/ReadPipeline.h"
#include "math_utils.h"

namespace dorado::utils {

void stitch_chunks(std::shared_ptr<Read> read) {
    // Calculate the chunk down sampling, round to closest int.
    read->model_stride = div_round_closest(read->called_chunks[0]->raw_chunk_size,
                                           read->called_chunks[0]->moves.size());

    int start_pos = 0;
    int mid_point_front = 0;
    std::vector<uint8_t> moves;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;

    for (int i = 0; i < read->num_chunks - 1; i++) {
        auto current_chunk = read->called_chunks[i];
        auto next_chunk = read->called_chunks[i + 1];
        int overlap_size = (current_chunk->raw_chunk_size + current_chunk->input_offset) -
                           (next_chunk->input_offset);
        assert(overlap_size % read->model_stride == 0);
        int overlap_down_sampled = overlap_size / read->model_stride;
        int mid_point_rear = overlap_down_sampled / 2;

        int current_chunk_bases_to_trim =
                std::accumulate(std::prev(current_chunk->moves.end(), mid_point_rear),
                                current_chunk->moves.end(), 0);

        int current_chunk_seq_len = current_chunk->seq.size();
        int end_pos = current_chunk_seq_len - current_chunk_bases_to_trim;
        int trimmed_len = end_pos - start_pos;
        sequences.push_back(current_chunk->seq.substr(start_pos, trimmed_len));
        qstrings.push_back(current_chunk->qstring.substr(start_pos, trimmed_len));
        moves.insert(moves.end(), std::next(current_chunk->moves.begin(), mid_point_front),
                     std::prev(current_chunk->moves.end(), mid_point_rear));

        mid_point_front = overlap_down_sampled - mid_point_rear;

        start_pos = 0;
        for (int i = 0; i < mid_point_front; i++) {
            start_pos += (int)next_chunk->moves[i];
        }
    }

    // Append the final chunk
    auto& last_chunk = read->called_chunks[read->num_chunks - 1];
    moves.insert(moves.end(), std::next(last_chunk->moves.begin(), mid_point_front),
                 last_chunk->moves.end());

    if (read->num_chunks == 1) {
        // shorten the sequence, qstring & moves where the read is shorter than chunksize
        int last_index_in_moves_to_keep = read->raw_data.size(0) / read->model_stride;
        moves = std::vector<uint8_t>(moves.begin(), moves.begin() + last_index_in_moves_to_keep);
        int end = std::accumulate(moves.begin(), moves.end(), 0);
        sequences.push_back(last_chunk->seq.substr(start_pos, end));
        qstrings.push_back(last_chunk->qstring.substr(start_pos, end));

    } else {
        sequences.push_back(last_chunk->seq.substr(start_pos));
        qstrings.push_back(last_chunk->qstring.substr(start_pos));
    }

    // Set the read seq and qstring
    read->seq = std::accumulate(sequences.begin(), sequences.end(), std::string(""));
    read->qstring = std::accumulate(qstrings.begin(), qstrings.end(), std::string(""));
    read->moves = std::move(moves);

    // remove partial stride overhang
    if (read->moves.size() > static_cast<int>(read->raw_data.size(0) / read->model_stride)) {
        if (read->moves.back() == 1) {
            read->seq.pop_back();
            read->qstring.pop_back();
        }
        read->moves.pop_back();
        assert(std::accumulate(read->moves.begin(), read->moves.end(), 0) == read->seq.size());
    }
}

}  // namespace dorado::utils
