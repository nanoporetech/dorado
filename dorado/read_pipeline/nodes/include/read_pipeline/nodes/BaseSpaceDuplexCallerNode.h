#pragma once

#include "read_pipeline/base/HtsReader.h"  // for ReadMap
#include "read_pipeline/base/MessageSink.h"

#include <map>
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
    ~BaseSpaceDuplexCallerNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    void start_threads();
    void terminate_impl(utils::AsyncQueueTerminateFast fast);
    void worker_thread();
    void basespace(const std::string& template_read_id, const std::string& complement_read_id);

    const size_t m_num_worker_threads;
    std::thread m_worker_thread;
    const std::map<std::string, std::string> m_template_complement_map;
    const ReadMap m_reads;
};
}  // namespace dorado
