#pragma once
#include "ReadPipeline.h"
#include "utils/bam_utils.h"

namespace dorado {
// Duplex caller node receives a map of template_id to complement_id (typically generated from a pairs file),
// and a map of `read_id` to `dorado::Read` object. It then performs duplex calling and pushes `dorado::Read`
// objects to its output queue.
class DuplexCallerNode : public ReadSink {
public:
    DuplexCallerNode(ReadSink& sink,
                     std::map<std::string, std::string> template_complement_map,
                     std::map<std::string, std::shared_ptr<Read>> reads);
    ~DuplexCallerNode();

private:
    void worker_thread();
    ReadSink&
            m_sink;  // ReadSink to consume Duplex Called Reads. This will typically be a writer node
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::shared_ptr<Read>> m_reads;
};
}  // namespace dorado
