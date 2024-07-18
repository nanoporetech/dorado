#include "ReadPipeline.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

using namespace std::chrono_literals;

namespace dorado {

// Depth first search that establishes a topological ordering for node destruction.
// Returns true if a cycle is found.
bool Pipeline::DFS(const std::vector<PipelineDescriptor::NodeDescriptor> &node_descriptors,
                   NodeHandle node_handle,
                   std::vector<DFSState> &dfs_state,
                   std::vector<NodeHandle> &source_to_sink_order) {
    auto &node_state = dfs_state.at(node_handle);
    if (node_state == DFSState::Visited) {
        // Already reached this node via another route.
        return false;
    }
    if (node_state == DFSState::Visiting) {
        // Back edge => cycle.
        return true;
    }
    node_state = DFSState::Visiting;
    auto sink_handles = node_descriptors.at(node_handle).sink_handles;
    for (auto sink_handle : sink_handles) {
        if (DFS(node_descriptors, sink_handle, dfs_state, source_to_sink_order)) {
            return true;
        }
    }
    node_state = DFSState::Visited;
    source_to_sink_order.push_back(node_handle);
    return false;
}

std::unique_ptr<Pipeline> Pipeline::create(
        PipelineDescriptor &&descriptor,
        std::vector<dorado::stats::StatsReporter> *const stats_reporters) {
    // Find a source node, i.e. one that is not the sink of any other node.
    // There should be exactly 1 one for a valid pipeline.
    const auto node_count = descriptor.m_node_descriptors.size();
    std::vector<bool> is_sink(node_count, false);
    for (auto &[desc_node, sink_handles] : descriptor.m_node_descriptors) {
        for (auto sink_handle : sink_handles) {
            is_sink.at(sink_handle) = true;
        }
    }
    const auto num_sources = std::count(is_sink.cbegin(), is_sink.cend(), false);
    if (num_sources != 1) {
        spdlog::error("There must be exactly 1 source node.  {} were present.", num_sources);
        return nullptr;
    }
    auto source_it = std::find(is_sink.cbegin(), is_sink.cend(), false);
    NodeHandle source_node = static_cast<NodeHandle>(std::distance(is_sink.cbegin(), source_it));

    // Perform a depth first search from the source to determine the
    // source-to-sink destruction order.  At the same time, cycles are detected.
    std::vector<NodeHandle> source_to_sink_order;
    std::vector<DFSState> dfs_state(node_count, DFSState::Unvisited);
    const bool has_cycle =
            DFS(descriptor.m_node_descriptors, source_node, dfs_state, source_to_sink_order);
    if (has_cycle) {
        spdlog::error("Graph has cycle");
        return nullptr;
    }
    std::reverse(source_to_sink_order.begin(), source_to_sink_order.end());
    // If the graph is fully connected then we should have visited all nodes.
    assert(std::all_of(dfs_state.cbegin(), dfs_state.cend(),
                       [](DFSState v) { return v == DFSState::Visited; }));
    assert(source_to_sink_order.size() == descriptor.m_node_descriptors.size());

    return std::unique_ptr<Pipeline>(
            new Pipeline(std::move(descriptor), source_to_sink_order, stats_reporters));
}

Pipeline::Pipeline(PipelineDescriptor &&descriptor,
                   std::vector<NodeHandle> source_to_sink_order,
                   std::vector<dorado::stats::StatsReporter> *const stats_reporters)
        : m_source_to_sink_order(std::move(source_to_sink_order)) {
    for (auto &[desc_node, _] : descriptor.m_node_descriptors) {
        m_nodes.push_back(std::move(desc_node));
    }

    if (stats_reporters) {
        for (auto node_index : m_source_to_sink_order) {
            stats_reporters->push_back(stats::make_stats_reporter(*m_nodes.at(node_index)));
        }
    }

    for (size_t i = 0; i < m_nodes.size(); ++i) {
        auto &node = m_nodes.at(i);
        const auto &sink_handles = descriptor.m_node_descriptors.at(i).sink_handles;
        for (const auto sink_handle : sink_handles) {
            node->add_sink(dynamic_cast<MessageSink &>(*m_nodes.at(sink_handle)));
        }
        // Start the node.
        node->restart();
    }
}

void Pipeline::push_message(Message &&message) {
    assert(!m_nodes.empty());
    const auto source_node_index = m_source_to_sink_order.front();
    dynamic_cast<MessageSink &>(*m_nodes.at(source_node_index)).push_message(std::move(message));
}

stats::NamedStats Pipeline::terminate(const FlushOptions &flush_options) {
    stats::NamedStats final_stats;
    // Nodes must be terminated in source to sink order to ensure all in flight
    // processing is completed, and sources still have valid sinks as they finish
    // work.
    for (auto handle : m_source_to_sink_order) {
        auto &node = m_nodes.at(handle);
        node->terminate(flush_options);
        auto node_stats = node->sample_stats();
        const auto node_name = node->get_name();
        for (const auto &[name, value] : node_stats) {
            final_stats[std::string(node_name).append(".").append(name)] = value;
        }
    }
    return final_stats;
}

void Pipeline::restart() {
    // The order in which we restart nodes shouldn't matter, so
    // we go source to sink.
    for (auto handle : m_source_to_sink_order) {
        m_nodes.at(handle)->restart();
    }
}

Pipeline::~Pipeline() {
    for (auto handle : m_source_to_sink_order) {
        auto &node = m_nodes.at(handle);
        node.reset();
    }
}

}  // namespace dorado
