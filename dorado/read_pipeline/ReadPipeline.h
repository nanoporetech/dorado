#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class Read;

struct Chunk {
    Chunk(std::shared_ptr<Read> read, size_t offset, size_t chunk_in_read_idx, size_t chunk_size)
            : source_read(read),
              input_offset(offset),
              idx_in_read(chunk_in_read_idx),
              raw_chunk_size(chunk_size){};

    std::weak_ptr<Read> source_read;
    size_t input_offset;    // Where does this chunk start in the input raw read data
    size_t idx_in_read;     // Just for tracking that the chunks don't go out of order
    size_t raw_chunk_size;  // Just for knowing the original chunk size

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;  // For stitching.
};

// Class representing a read, including raw data
class Read {
public:
    struct Attributes {
        uint32_t mux{std::numeric_limits<uint32_t>::max()};  // Channel mux
        uint32_t read_number{std::numeric_limits<
                uint32_t>::max()};  // Per-channel number of each read as it was acquired by minknow
        int32_t channel_number{-1};  //Channel ID
        std::string start_time{};    //Read acquisition start time
        std::string fast5_filename{};
    };

    struct Mapping {
        // Dummy struct for future use to represent alignments
    };

    torch::Tensor raw_data;  // Loaded from source file
    float digitisation;      // Loaded from source file
    float range;             // Loaded from source file
    float offset;            // Loaded from source file

    float med;  // To be set by scaler
    float mad;  // To be set by scaler

    bool scale_set = false;  //Set to True if scale has been applied to raw data
    float scale;  // Scale factor applied to convert raw integers from sequencer into pore current values

    size_t num_chunks;  //Number of chunks in the read. Reads raw data is split into chunks for efficient basecalling.
    std::vector<std::shared_ptr<Chunk>> called_chunks;  // Vector of basecalled chunks.
    std::atomic_size_t num_chunks_called;  // Number of chunks which have been basecalled

    std::string read_id;  //Unique read ID (UUID4)
    std::string seq;      //Read basecall
    std::string qstring;  //Read Qstring

    uint64_t num_samples;          //Number of raw samples in read
    uint64_t num_trimmed_samples;  //Number of samples which have been trimmed from the raw read.

    Attributes attributes;
    std::vector<Mapping> mappings;

    std::vector<std::string> generate_read_tags() const;
    std::vector<std::string> extract_sam_lines() const;
};

// Base class for an object which consumes reads.
// ReadSink is a node within a pipeline.
class ReadSink {
public:
    ReadSink(size_t max_reads);
    void push_read(
            std::shared_ptr<Read>&
                    read);  //Push a read into readsink. This can block if receiving ReadSink is full.
    void terminate() { m_terminate = true; }  // Notify sinks and terminate.
protected:
    std::condition_variable m_cv;
    std::mutex m_cv_mutex;
    std::condition_variable m_push_read_cv;
    std::mutex m_push_read_cv_mutex;

    size_t m_max_reads =
            1000;  //ReadSink will block on accepting reads if it contains m_max_reads reads
    bool m_terminate = false;  // When set to true, ReadSink will notify it's sinks and terminate.

    // The queue of reads itself
    std::deque<std::shared_ptr<Read>> m_reads;
};
