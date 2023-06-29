#pragma once
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace dorado {

namespace utils {
struct BaseModInfo;
}

class Read;

struct Chunk {
    Chunk(std::shared_ptr<Read> const& read,
          size_t offset,
          size_t chunk_in_read_idx,
          size_t chunk_size)
            : source_read(read),
              input_offset(offset),
              idx_in_read(chunk_in_read_idx),
              raw_chunk_size(chunk_size) {}

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
        int32_t read_number{-1};  // Per-channel number of each read as it was acquired by minknow
        int32_t channel_number{-1};  //Channel ID
        std::string start_time{};    //Read acquisition start time
        std::string fast5_filename{};
        uint64_t num_samples;
    };

    struct Mapping {
        // Dummy struct for future use to represent alignments
    };

    torch::Tensor raw_data;  // Loaded from source file
    float digitisation;      // Loaded from source file
    float range;             // Loaded from source file
    float offset;            // Loaded from source file

    uint64_t sample_rate;  // Loaded from source file

    uint64_t start_time_ms;
    uint64_t get_end_time_ms();

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
    std::string run_id;                   // Run ID - used in read group
    std::string flowcell_id;              // Flowcell ID - used in read group
    std::string model_name;               // Read group

    std::string parent_read_id;  // Origin read ID for all its subreads. Empty for nonsplit reads.

    std::shared_ptr<const utils::BaseModInfo>
            base_mod_info;  // Modified base settings of the models that ran on this read

    uint64_t num_trimmed_samples;  // Number of samples which have been trimmed from the raw read.

    Attributes attributes;
    std::vector<Mapping> mappings;
    std::vector<BamPtr> extract_sam_lines(bool emit_moves, uint8_t modbase_threshold = 0) const;

    uint64_t start_sample;
    uint64_t end_sample;
    uint64_t run_acquisition_start_time_ms;
    bool is_duplex;
    std::atomic_size_t num_duplex_candidate_pairs{0};

    size_t subread_id{0};
    size_t split_count{1};

    // A unique identifier for each input read
    // Split (duplex) reads have the read_tag of the parent (template) and their own subread_id
    uint64_t read_tag{0};
    // The id of the client to which this read belongs. -1 in standalone mode
    int32_t client_id{-1};

private:
    void generate_duplex_read_tags(bam1_t*) const;
    void generate_read_tags(bam1_t* aln, bool emit_moves) const;
    void generate_modbase_string(bam1_t* aln, uint8_t threshold = 0) const;
    std::string generate_read_group() const;
};

// A pair of reads for Duplex calling
class ReadPair {
public:
    std::shared_ptr<Read> read_1;
    std::shared_ptr<Read> read_2;
};

class CandidatePairRejectedMessage {};

// The Message type is a std::variant that can hold different types of message objects.
// It is currently able to store:
// - a std::shared_ptr<Read> object, which represents a single read
// - a BamPtr object, which represents a raw BAM alignment record
// - a std::shared_ptr<ReadPair> object, which represents a pair of reads for duplex calling
// - a std::shared_ptr<CandidatePairRejectedMessage> object, which informs downstream processing that a candidate pair has been rejected
// To add more message types, simply add them to the list of types in the std::variant.
using Message = std::variant<std::shared_ptr<Read>,
                             BamPtr,
                             std::shared_ptr<ReadPair>,
                             CandidatePairRejectedMessage>;

// Base class for an object which consumes messages.
// MessageSink is a node within a pipeline.
// NOTE: In order to prevent potential deadlocks when
// the writer to the node doesn't exit cleanly, always
// call terminate() in the destructor of a class derived
// from MessageSink (and before worker thread join() calls
// if there are any).
class MessageSink {
public:
    MessageSink(size_t max_messages);
    virtual ~MessageSink() = default;
    // Pushed messages must be rvalues: the sink takes ownership.
    void push_message(
            Message&&
                    message);  // Push a message into message sink.  This can block if the sink's queue is full.
    void terminate() { m_work_queue.terminate(); }

    // StatsSampler will ignore nodes with an empty name.
    virtual std::string get_name() const { return std::string(""); }
    virtual stats::NamedStats sample_stats() const {
        return std::unordered_map<std::string, double>();
    }

protected:
    // Queue of work items for this node.
    AsyncQueue<Message> m_work_queue;
};

}  // namespace dorado
