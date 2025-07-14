#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <cstddef>
#include <string>

namespace dorado {

namespace hts_writer {
class IWriter;
}

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

    // Set the header for all hts file writers
    void set_hts_file_header(SamHdrPtr hdr) const;

private:
    void input_thread_fn();

    const std::vector<std::unique_ptr<hts_writer::IWriter>> m_writers;
    const size_t m_num_writers{0};
};

}  // namespace dorado
