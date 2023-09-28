#pragma once

#include <memory>
#include <string>
#include <vector>

namespace dorado {
class ReadCommon;
}  // namespace dorado

namespace dorado::utils {

// A single chunk
struct Chunk {
    Chunk(size_t offset, size_t chunk_size) : input_offset(offset), raw_chunk_size(chunk_size) {}
    virtual ~Chunk() = default;

    size_t input_offset;    // Where does this chunk start in the input raw read data
    size_t raw_chunk_size;  // Just for knowing the original chunk size

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;  // For stitching.
};

// Given a read and its unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and
// qstring to Read
void stitch_chunks(ReadCommon& read, const std::vector<std::unique_ptr<Chunk>>& called_chunks);

}  // namespace dorado::utils
