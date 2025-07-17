#include "hts_writer/StreamHtsFileWriter.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace dorado {
namespace hts_writer {

StreamHtsFileWriter::StreamHtsFileWriter(const HtsFileWriterConfig& cfg)
        : HtsFileWriter(
                  {cfg.mode, 0, cfg.progress_callback, cfg.description_callback, cfg.gpu_names}) {}

void StreamHtsFileWriter::shutdown() {
    if (std::exchange(m_has_shutdown, true)) {
        return;
    }

    if (m_hts_file == nullptr || m_hts_file.get() == nullptr) {
        spdlog::debug(
                "StreamHtsFileWriter::shutdown called on uninitialised hts_file - nothing to do");
        return;
    }
    set_description("Finalising output");
    m_hts_file->finalise([this](size_t progress) { set_progress(progress); });
}

bool StreamHtsFileWriter::finalise_is_noop() const { return true; };

void StreamHtsFileWriter::handle(const HtsData& data) {
    if (m_has_shutdown) {
        spdlog::debug("HtsFileWriter has shutdown and cannot process more work.");
        return;
    }

    if (m_hts_file == nullptr) {
        if (m_header == nullptr) {
            throw std::logic_error("HtsFileWriter header not set before writing records.");
        }
        m_hts_file = std::make_unique<utils::HtsFile>(m_path, m_mode, 0, false);
        m_hts_file->set_header(m_header.get());
    }
    m_hts_file->write(data.bam_ptr.get());
}

}  // namespace hts_writer
}  // namespace dorado
