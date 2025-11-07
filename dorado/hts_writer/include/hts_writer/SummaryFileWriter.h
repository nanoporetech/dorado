#pragma once

#include "interface.h"

#include <filesystem>
#include <fstream>
#include <memory>

namespace dorado {
class HtsData;

namespace hts_writer {
class SummaryFileWriter : public IWriter {
public:
    SummaryFileWriter(const std::filesystem::path& output_directory);
    SummaryFileWriter(std::ostream& stream);

    void process(const Processable item) override;
    void shutdown() override;

    std::string get_name() const override { return "SummaryFileWriter"; }
    stats::NamedStats sample_stats() const override { return {}; }

private:
    void init();
    void handle(const HtsData& item);

    std::ofstream m_summary_file;
    std::ostream& m_summary_stream;
};

}  // namespace hts_writer
}  // namespace dorado
