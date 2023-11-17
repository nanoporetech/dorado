#pragma once

#include "ReadPipeline.h"

namespace dorado {

class NullNode : public MessageSink {
public:
    // NullNode has no sink - input messages go nowhere
    NullNode();
    ~NullNode() { terminate_impl(); }
    std::string get_name() const override { return "NullNode"; }
    void terminate(const FlushOptions &) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();
    std::vector<std::unique_ptr<std::thread>> m_workers;
};

}  // namespace dorado
