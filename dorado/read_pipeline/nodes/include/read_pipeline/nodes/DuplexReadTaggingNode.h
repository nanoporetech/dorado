#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado {

/// Class to correctly tag the duplex
/// and simplex reads based on, post filtering,
/// whether a simplex read is a parent of a duplex
/// read or not.
class DuplexReadTaggingNode final : public MessageSink {
public:
    DuplexReadTaggingNode();
    ~DuplexReadTaggingNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    DuplexReadTaggingNode(const DuplexReadTaggingNode&) = delete;
    DuplexReadTaggingNode& operator=(const DuplexReadTaggingNode&) = delete;
    DuplexReadTaggingNode(DuplexReadTaggingNode&&) = delete;
    DuplexReadTaggingNode& operator=(DuplexReadTaggingNode&&) = delete;

private:
    void input_thread_fn();

    std::unordered_map<std::string, SimplexReadPtr> m_duplex_parents;
    std::unordered_set<std::string> m_parents_processed;
    std::unordered_set<std::string> m_parents_wanted;
};

}  // namespace dorado
