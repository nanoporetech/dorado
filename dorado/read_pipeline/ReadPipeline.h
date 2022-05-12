#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

class Read;

struct Chunk {
    Chunk(std::shared_ptr<Read> read, size_t offset, size_t chunk_in_read_idx, size_t chunk_size) :
        source_read(read),
        input_offset(offset),
        idx_in_read(chunk_in_read_idx),
        raw_chunk_size(chunk_size){};

    std::weak_ptr<Read> source_read;
    size_t input_offset; // Where does this chunk start in the input raw read data
    size_t idx_in_read; // Just for tracking that the chunks don't go out of order
    size_t raw_chunk_size; // Just for knowing the original chunk size

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves; // For stitching.
};

// Object representing a simplex read
class Read {
public:
    struct Attributes {
        uint32_t mux{std::numeric_limits<uint32_t>::max()};
        uint32_t read_number{std::numeric_limits<uint32_t>::max()};
        int32_t channel_number{-1};
        std::string start_time{};
        std::string fast5_filename{};
    };

    struct Mapping {
        // Dummy struct for future use to represent alignments
    };

    torch::Tensor raw_data; // Loaded from source file
    float digitisation; // Loaded from source file
    float range; // Loaded from source file
    float offset; // Loaded from source file

    float med; // To be set by scaler
    float mad; // To be set by scaler

    size_t num_chunks;
    std::vector<std::shared_ptr<Chunk>> called_chunks;
    std::atomic_size_t num_chunks_called;

    std::string read_id;
    std::string seq;
    std::string qstring;

    uint64_t num_samples;
    uint64_t num_trimmed_samples;

    Attributes attributes;
    std::vector<Mapping> mappings;

    std::vector<std::string> generate_read_tags() const;
    std::vector<std::string> extract_sam_lines() const;
};


// Base class for an object which consumes reads
class ReadSink {
public:
    ReadSink(size_t max_reads);
    void push_read(std::shared_ptr<Read>& read);
    void terminate() { m_terminate = true; }
protected:
    std::condition_variable m_cv;
    std::mutex m_cv_mutex;
    std::condition_variable m_push_read_cv;
    std::mutex m_push_read_cv_mutex;

    size_t m_max_reads = 1000;
    bool m_terminate = false;

    // The queue of reads itself
    std::deque<std::shared_ptr<Read>> m_reads;
};
