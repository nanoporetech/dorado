#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace utils {
struct BaseModInfo;
}

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
    float sample_rate;       // Loaded from source file

    float shift;  // To be set by scaler
    float scale;  // To be set by scaler

    float scaling;  // Scale factor applied to convert raw integers from sequencer into pore current values

    size_t num_chunks;  // Number of chunks in the read. Reads raw data is split into chunks for efficient basecalling.
    std::vector<std::shared_ptr<Chunk>> called_chunks;  // Vector of basecalled chunks.
    std::atomic_size_t num_chunks_called;  // Number of chunks which have been basecalled

    size_t num_modbase_chunks;
    std::atomic_size_t
            num_modbase_chunks_called;  // Number of modbase chunks which have been scored

    int model_stride;  // The down sampling factor of the model

    std::string read_id;                  // Unique read ID (UUID4)
    std::string seq;                      // Read basecall
    std::string qstring;                  // Read Qstring (Phred)
    std::vector<uint8_t> moves;           // Move table
    std::vector<uint8_t> base_mod_probs;  // Modified base probabilities
    std::string run_id;                   // Read group
    std::string model_name;               // Read group

    std::shared_ptr<const utils::BaseModInfo>
            base_mod_info;  // Modified base settings of the models that ran on this read

    uint64_t num_trimmed_samples;  // Number of samples which have been trimmed from the raw read.

    Attributes attributes;
    std::vector<Mapping> mappings;
    std::vector<std::string> generate_duplex_read_tags() const;
    std::vector<std::string> generate_read_tags(bool emit_moves) const;
    std::vector<std::string> extract_sam_lines(bool emit_moves, bool duplex) const;
    std::string generate_modbase_string(uint8_t threshold = 0) const;
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

}  // namespace dorado
