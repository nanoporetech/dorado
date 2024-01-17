#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado {

/// Class to correctly tag the duplex
/// and simplex reads based on, post filtering,
/// whether a simplex read is a parent of a duplex
/// read or not.
class DuplexReadTaggingNode : public MessageSink {
public:
    DuplexReadTaggingNode();
    ~DuplexReadTaggingNode() { stop_input_processing(); }
    std::string get_name() const override { return "DuplexReadTaggingNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override;

private:
    void input_thread_fn();

    std::unordered_map<std::string, SimplexReadPtr> m_duplex_parents;
    std::unordered_set<std::string> m_parents_processed;
    std::unordered_set<std::string> m_parents_wanted;
};

}  // namespace dorado
