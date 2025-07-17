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
    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &terminate_options) override;
    void restart() override;

    // Set the header for all hts file writers
    void set_hts_file_header(SamHdrPtr hdr) const;

private:
    void input_thread_fn();

    const std::vector<std::unique_ptr<hts_writer::IWriter>> m_writers;
};

}  // namespace dorado
