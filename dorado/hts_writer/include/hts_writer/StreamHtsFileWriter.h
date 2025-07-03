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
    StreamHtsFileWriter(OutputMode mode,
                        utils::ProgressCallback progress_callback,
                        utils::DescriptionCallback description_callback);
    void init() override;
    void shutdown() override;

    bool finalise_is_noop() const override { return true; };

private:
    std::unique_ptr<utils::HtsFile> m_hts_file;

    void handle(const HtsData& data) override;
};

}  // namespace hts_writer
}  // namespace dorado
