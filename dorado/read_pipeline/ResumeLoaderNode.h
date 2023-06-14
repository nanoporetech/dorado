#pragma once

#include "ReadPipeline.h"

#include <string>
#include <unordered_set>

namespace dorado {

/// Class to filter reads based on some criteria.
/// Currently only supports filtering based on
/// minimum Q-score, read length and read id.
class ResumeLoaderNode {
public:
    ResumeLoaderNode(MessageSink& sink, const std::string& resume_file);
    ~ResumeLoaderNode() = default;
    void copy_completed_reads();
    std::unordered_set<std::string> get_processed_read_ids() const;

private:
    MessageSink& m_sink;
    std::string m_resume_file;

    std::unordered_set<std::string> m_processed_read_ids;
};

}  // namespace dorado
