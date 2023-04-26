#pragma once

#include "ReadPipeline.h"
#include "data_loader/DataLoader.h"

#include <indicators/progress_bar.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace dorado {

class NullNode : public MessageSink {
public:
    // NullNode has no sink - input messages go nowhere
    NullNode();
    ~NullNode();

private:
    void worker_thread();
    std::vector<std::unique_ptr<std::thread>> m_workers;
};

}  // namespace dorado
