#pragma once

#include "MessageSink.h"

#include <string>
#include <unordered_set>

namespace dorado {

class ResumeLoader {
public:
    ResumeLoader(MessageSink& sink, const std::string& resume_file);

    void copy_completed_reads();
    std::unordered_set<std::string> get_processed_read_ids() const;

private:
    MessageSink& m_sink;
    std::string m_resume_file;

    std::unordered_set<std::string> m_processed_read_ids;
};

}  // namespace dorado
