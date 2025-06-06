#pragma once

#include "MessageSink.h"
#include "messages.h"
#include "utils/stats.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

using NodeHandle = int;

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
        NodeDescriptor node_desc;
        node_desc.node = std::move(node);
        node_desc.sink_handles = std::move(sink_handles);
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
    static std::unique_ptr<Pipeline> create(PipelineDescriptor&& descriptor,
                                            std::vector<stats::StatsReporter>* stats_reporters);

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
    template <typename NodeType>
    NodeType& get_node_ref(NodeHandle node_handle) {
        // .at() will throw if the index is bad.
        // dynamic_cast<&> will throw if this isn't the right type.
        return dynamic_cast<NodeType&>(*m_nodes.at(node_handle));
    }

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
