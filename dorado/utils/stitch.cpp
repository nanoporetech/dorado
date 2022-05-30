#include "../read_pipeline/ReadPipeline.h"

int div_round_closest(const int n, const int d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

void stitch_chunks(std::shared_ptr<Read> read) {
    //Calculate the chunk down sampling, round to closest int.
    int down_sampling = div_round_closest(read->called_chunks[0]->raw_chunk_size,
                                          read->called_chunks[0]->moves.size());

    std::vector<uint8_t> moves = read->called_chunks[0]->moves;

    int start_pos = 0;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;
    for (int i = 0; i < read->num_chunks - 1; i++) {
        auto current_chunk = read->called_chunks[i];
        auto next_chunk = read->called_chunks[i + 1];
        int overlap_size = (current_chunk->raw_chunk_size + current_chunk->input_offset) -
                           (next_chunk->input_offset);
        int overlap_down_sampled = overlap_size / down_sampling;
        int mid_point = overlap_down_sampled / 2;

        int current_chunk_bases_to_trim = 0;
        for (int i = current_chunk->moves.size() - 1; i > current_chunk->moves.size() - mid_point;
             i--) {
            current_chunk_bases_to_trim += (int)current_chunk->moves[i];
        }

        int current_chunk_seq_len = current_chunk->seq.size();
        int end_pos = current_chunk_seq_len - current_chunk_bases_to_trim;
        int trimmed_len = end_pos - start_pos;
        sequences.push_back(current_chunk->seq.substr(start_pos, trimmed_len));
        qstrings.push_back(current_chunk->qstring.substr(start_pos, trimmed_len));

        start_pos = 0;
        for (int i = 0; i < mid_point; i++) {
            start_pos += (int)next_chunk->moves[i];
        }
    }

    //append the final read
    sequences.push_back(read->called_chunks[read->num_chunks - 1]->seq.substr(start_pos));
    qstrings.push_back(read->called_chunks[read->num_chunks - 1]->qstring.substr(start_pos));

    // Set the read seq and qstring
    read->seq = std::accumulate(sequences.begin(), sequences.end(), std::string(""));
    read->qstring = std::accumulate(qstrings.begin(), qstrings.end(), std::string(""));
}
