#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#include <cstdint>
#include <string>

namespace dorado {

class PafWriter : public MessageSink {
public:
    PafWriter();
    ~PafWriter();
    std::string get_name() const override { return "PafWriter"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override;
    void restart() override { start_input_processing(&PafWriter::input_thread_fn, this); }

private:
    void input_thread_fn();
};

}  // namespace dorado
