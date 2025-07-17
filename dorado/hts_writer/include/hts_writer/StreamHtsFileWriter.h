#pragma once

#include "hts_writer/HtsFileWriter.h"

#include <memory>

namespace dorado {

namespace utils {
class HtsFile;
}

namespace hts_writer {

using OutputMode = utils::HtsFile::OutputMode;

class StreamHtsFileWriter : public HtsFileWriter {
public:
    StreamHtsFileWriter(const HtsFileWriterConfig& cfg);
    void shutdown() override;
    bool finalise_is_noop() const override;

private:
    std::unique_ptr<utils::HtsFile> m_hts_file;
    const std::string m_path{"-"};

    bool m_has_shutdown{false};

    void handle(const HtsData& data) override;
};

}  // namespace hts_writer
}  // namespace dorado
