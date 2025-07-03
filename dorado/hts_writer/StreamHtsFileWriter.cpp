#include "hts_writer/StreamHtsFileWriter.h"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace dorado {
namespace hts_writer {

StreamHtsFileWriter::StreamHtsFileWriter(OutputMode mode,
                                         utils::ProgressCallback progress_callback,
                                         utils::DescriptionCallback description_callback)
        : HtsFileWriter(mode, 0, std::move(progress_callback), std::move(description_callback)) {}

void StreamHtsFileWriter::init() {}

void StreamHtsFileWriter::shutdown() {
    set_description("Finalising output");
    m_hts_file->finalise([this](size_t progress) { set_progress(progress); });
}

void StreamHtsFileWriter::handle(const HtsData &data) {
    if (m_hts_file == nullptr) {
        m_hts_file = std::make_unique<utils::HtsFile>("-", m_mode, 0, false);
        if (m_hts_file == nullptr) {
            std::runtime_error("Failed to create HTS output file");
        }
        if (m_header == nullptr) {
            std::logic_error("HtsFileWriter header not set before writing records.");
        }
        m_hts_file->set_header(m_header.get());
    }
    m_hts_file->write(data.bam_ptr.get());
}

}  // namespace hts_writer
}  // namespace dorado
