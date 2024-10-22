#pragma once

#include "HtsReader.h"
#include "MessageSink.h"
#include "utils/bam_utils.h"

#include <map>
#include <memory>
#include <string>

namespace dorado {
// Duplex caller node receives a map of template_id to complement_id (typically generated from a pairs file),
// and a map of `read_id` to `dorado::Read` object. It then performs duplex calling and pushes `dorado::Read`
// objects to its output queue.
class BaseSpaceDuplexCallerNode : public MessageSink {
public:
    BaseSpaceDuplexCallerNode(std::map<std::string, std::string> template_complement_map,
                              ReadMap reads,
                              size_t threads);
    ~BaseSpaceDuplexCallerNode() { terminate_impl(); }
    std::string get_name() const override { return "BaseSpaceDuplexCallerNode"; }
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();
    void basespace(const std::string& template_read_id, const std::string& complement_read_id);

    const size_t m_num_worker_threads;
    std::thread m_worker_thread;
    std::map<std::string, std::string> m_template_complement_map;
    const ReadMap m_reads;
};
}  // namespace dorado
