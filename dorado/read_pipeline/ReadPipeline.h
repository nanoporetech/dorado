#pragma once
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace dorado {

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

    float shift;                 // To be set by scaler
    float scale;                 // To be set by scaler
    std::string scaling_method;  // To be set by scaler

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

    std::shared_ptr<const ModBaseInfo>
            mod_base_info;  // Modified base settings of the models that ran on this read

    uint64_t num_trimmed_samples;  // Number of samples which have been trimmed from the raw read.

    Attributes attributes;
    std::vector<Mapping> mappings;
    std::vector<BamPtr> extract_sam_lines(bool emit_moves, uint8_t modbase_threshold = 0) const;

    float calculate_mean_qscore() const;

    uint64_t start_sample;
    uint64_t end_sample;
    uint64_t run_acquisition_start_time_ms;
    bool is_duplex{false};
    bool is_duplex_parent{false};
    // Calculate mean Q-score from this position onwards if read is
    // a short read.
    uint32_t mean_qscore_start_pos = 0;

    std::atomic_size_t num_duplex_candidate_pairs{0};

    // subread_id is used to track 2 types of offsprings of a read
    // (1) read splits
    // (2) duplex pairs which share this read as the template read
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
    uint64_t read_1_start;
    uint64_t read_1_end;
    uint64_t read_2_start;
    uint64_t read_2_end;
};

class CacheFlushMessage {
public:
    int32_t client_id;
};

// The Message type is a std::variant that can hold different types of message objects.
// It is currently able to store:
// - a std::shared_ptr<Read> object, which represents a single read
// - a BamPtr object, which represents a raw BAM alignment record
// - a std::shared_ptr<ReadPair> object, which represents a pair of reads for duplex calling
// To add more message types, simply add them to the list of types in the std::variant.
using Message =
        std::variant<std::shared_ptr<Read>, BamPtr, std::shared_ptr<ReadPair>, CacheFlushMessage>;

using NodeHandle = int;

struct FlushOptions {
    bool preserve_pairing_caches = false;
};
inline FlushOptions DefaultFlushOptions() { return {false}; }

// Base class for an object which consumes messages as part of the processing pipeline.
// Destructors of derived classes must call terminate() in order to shut down
// waits on the input queue before attempting to join input worker threads.
class MessageSink {
public:
    MessageSink(size_t max_messages);
    virtual ~MessageSink() = default;

    // StatsSampler will ignore nodes with an empty name.
    virtual std::string get_name() const { return std::string(""); }
    virtual stats::NamedStats sample_stats() const {
        return std::unordered_map<std::string, double>();
    }

    // Adds a message to the input queue.  This can block if the sink's queue is full.
    template <typename Msg>
    void push_message(Msg&& msg) {
        static_assert(!std::is_reference_v<Msg> && !std::is_const_v<Msg>,
                      "Pushed messages must be rvalues: the sink takes ownership");
        push_message_internal(Message(std::move(msg)));
    }

    // Waits until work is finished and shuts down worker threads.
    // No work can be done by the node after this returns until
    // restart is subsequently called.
    virtual void terminate(const FlushOptions& flush_options) = 0;

    // Restarts the node following a terminate call.
    // Has no effect if terminate has not been called.
    virtual void restart() = 0;

protected:
    // Terminates waits on the input queue.
    void terminate_input_queue() { m_work_queue.terminate(); }

    // Allows inputs again.
    void restart_input_queue() { m_work_queue.restart(); }

    // Sends message to the designated sink.
    template <typename Msg>
    void send_message_to_sink(int sink_index, Msg&& message) {
        m_sinks.at(sink_index).get().push_message(std::forward<Msg>(message));
    }

    // Version for nodes with a single sink that is implicit.
    template <typename Msg>
    void send_message_to_sink(Msg&& message) {
        assert(m_sinks.size() == 1);
        send_message_to_sink(0, std::forward<Msg>(message));
    }

    // Pops the next input message, returning true on success.
    // If terminating, returns false.
    bool get_input_message(Message& message) {
        auto status = m_work_queue.try_pop(message);
        return status == utils::AsyncQueueStatus::Success;
    }

    // Queue of work items for this node.
    utils::AsyncQueue<Message> m_work_queue;

private:
    // The sinks to which this node can send messages.
    std::vector<std::reference_wrapper<MessageSink>> m_sinks;

    friend class Pipeline;
    void add_sink(MessageSink& sink);

    void push_message_internal(Message&& message);
};

// Object from which a Pipeline is created.
// While this currently embodies constructed pipeline nodes, the intent is that this
// could be evolve toward, or be augmented by, other means of describing the pipeline.
class PipelineDescriptor {
    friend class Pipeline;

    std::vector<std::unique_ptr<MessageSink>> m_nodes;
    std::vector<std::vector<NodeHandle>> m_node_sink_handles;

    struct NodeDescriptor {
        std::unique_ptr<MessageSink> node;
        std::vector<NodeHandle> sink_handles;
    };
    std::vector<NodeDescriptor> m_node_descriptors;

    bool is_handle_valid(NodeHandle handle) const {
        return handle >= 0 && handle < static_cast<int>(m_node_descriptors.size());
    }

public:
    static const NodeHandle InvalidNodeHandle = -1;

    // Adds the node of specified type, returning a handle.
    // 0 or more sinks can be specified here, and augmented subsequently via AddNodeSink.
    template <class NodeType, class... Args>
    NodeHandle add_node(std::vector<NodeHandle> sink_handles, Args&&... args) {
        // TODO -- probably want to make node constructors private, which would entail
        // avoiding make_unique.
        auto node = std::make_unique<NodeType>(std::forward<Args>(args)...);
        NodeDescriptor node_desc{std::move(node), std::move(sink_handles)};
        m_node_descriptors.push_back(std::move(node_desc));
        return static_cast<NodeHandle>(m_node_descriptors.size() - 1);
    }

    // Adds a sink the specified node.
    // Returns true on success.
    bool add_node_sink(NodeHandle node_handle, NodeHandle sink_handle) {
        if (!is_handle_valid(node_handle) || !is_handle_valid(sink_handle)) {
            spdlog::error("Invalid node handle");
            return false;
        }
        m_node_descriptors[node_handle].sink_handles.push_back(sink_handle);
        return true;
    }
};

// Created from PipelineDescriptor.  Accepts messages and processes them.
// When the Pipeline object is destroyed, the nodes it owns will be destroyed
// in an ordering where sources come before sinks.
class Pipeline {
public:
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&&) = default;
    Pipeline& operator=(Pipeline&&) = default;

    // Factory method that creates a Pipeline from a PipelineDescriptor, which is
    // consumed during creation.
    // If non-null, stats_reporters has node stats reporters added to it.
    // Returns the resulting pipeline, or a null unique_ptr on error.
    static std::unique_ptr<Pipeline> create(
            PipelineDescriptor&& descriptor,
            std::vector<stats::StatsReporter>* stats_reporters = nullptr,
            stats::NamedStats* final_stats = nullptr);

    // Routes the given message to the pipeline source node.
    void push_message(Message&& message);

    // Stops all pipeline nodes in source to sink order.
    // Returns stats from nodes' final states.
    // After this is called the pipeline will do no further work processing subsequent inputs,
    // unless restart is called first.
    stats::NamedStats terminate(const FlushOptions& flush_options);

    // Restarts pipeline after a call to terminate.
    void restart();

    // Returns a reference to the node associated with the given handle.
    // Exists to accommodate situations where client code avoids using the pipeline framework.
    MessageSink& get_node_ref(NodeHandle node_handle) { return *m_nodes.at(node_handle); }

private:
    // Constructor is private to ensure instances of this class are created
    // through the create function.
    Pipeline(PipelineDescriptor&& descriptor,
             std::vector<NodeHandle> source_to_sink_order,
             std::vector<dorado::stats::StatsReporter>* stats_reporters);

    std::vector<std::unique_ptr<MessageSink>> m_nodes;
    std::vector<NodeHandle> m_source_to_sink_order;

    enum class DFSState { Unvisited, Visiting, Visited };

    static bool DFS(const std::vector<PipelineDescriptor::NodeDescriptor>& node_descriptors,
                    NodeHandle node_handle,
                    std::vector<DFSState>& dfs_state,
                    std::vector<NodeHandle>& source_to_sink_order);
};

}  // namespace dorado
