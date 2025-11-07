#include "hts_writer/SummaryFileWriter.h"

#include "hts_utils/hts_types.h"

namespace dorado::hts_writer {

SummaryFileWriter::SummaryFileWriter(const std::filesystem::path& output_directory)
        : m_summary_file(output_directory / "sequencing_summary.txt"),
          m_summary_stream(m_summary_file) {
    init();
}

SummaryFileWriter::SummaryFileWriter(std::ostream& stream) : m_summary_stream(stream) { init(); }

void SummaryFileWriter::init() {
    // Write column headers
}

void SummaryFileWriter::process(const Processable item) {
    dispatch_processable(item, [this](const auto& t) { this->handle(t); });
}

void SummaryFileWriter::handle(const HtsData& data) {
    // skip secondary and supplementary alignments in the summary
    if (data.bam_ptr->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
        return;
    }

    // write column data
}

void SummaryFileWriter::shutdown() {
    // close the file if we're working with a file path
    if (m_summary_file.is_open()) {
        m_summary_file.close();
    }
}

}  // namespace dorado::hts_writer
