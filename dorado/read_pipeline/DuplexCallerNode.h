#pragma once
#include "ReadPipeline.h"
#include "utils/bam_utils.h"

namespace dorado {
class DuplexCallerNode : public ReadSink {
public:
    DuplexCallerNode(ReadSink& sink,
                     std::map<std::string, std::string>,
                     std::map<std::string, std::shared_ptr<Read>> reads);
    ~DuplexCallerNode();

private:
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    ReadSink&
            m_sink;  // ReadSink to consume Duplex Called Reads. This will typically be a writer node
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::shared_ptr<Read>> m_reads;
};
}