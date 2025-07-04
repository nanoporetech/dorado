#pragma once

#include "hts_utils/hts_types.h"
#include "hts_writer/interface.h"
#include "read_pipeline/base/MessageSink.h"

#include <string>

namespace dorado {

class WriterNode : public MessageSink {
public:
    WriterNode(std::vector<std::unique_ptr<hts_writer::IWriter>> writers);
    ~WriterNode();
    std::string get_name() const override { return "WriterNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "writer_node");
    }

    // Set the header for all owned HtsFileWriters
    void take_hts_header(SamHdrPtr hdr) const;

private:
    void input_thread_fn();

    std::atomic<int> m_num_received{0}, m_num_dispatched{0};
    const std::vector<std::unique_ptr<hts_writer::IWriter>> m_writers;
};

}  // namespace dorado
