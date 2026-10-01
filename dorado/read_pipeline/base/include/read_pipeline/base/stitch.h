#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace dorado {
class ReadCommon;
}  // namespace dorado

namespace dorado::utils {

// A single chunk
struct Chunk {
    // Non-VCS granularity is chunk_size
    Chunk(size_t offset, size_t chunk_size) : Chunk(offset, chunk_size, chunk_size) {}

    Chunk(size_t offset, size_t chunk_size, size_t chunk_granularity)
            : input_offset(offset),
              raw_chunk_size(chunk_size),
              chunk_size_granularity(chunk_granularity) {}

    size_t input_offset;    // Where does this chunk start in the input raw read data
    size_t raw_chunk_size;  // Just for knowing the original chunk size
    size_t chunk_size_granularity;  // For stitching, raw_chunk_size is padded to granularity when basecalling

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;  // For stitching.
    std::vector<uint8_t> base_mod_probs;
    std::vector<bool> base_mod_simplex_motif_hits;
};

// Given a read and its unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and
// qstring to Read
void stitch_chunks(ReadCommon& read, std::span<const Chunk*> called_chunks, bool is_vcs);

}  // namespace dorado::utils
