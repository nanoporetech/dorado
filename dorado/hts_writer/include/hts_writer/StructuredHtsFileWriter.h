#pragma once

#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/Structure.h"

#include <memory>
#include <unordered_map>

namespace dorado {

namespace hts_writer {

class StructuredHtsFileWriter : public HtsFileWriter {
public:
    StructuredHtsFileWriter(const HtsFileWriterConfig& cfg,
                            std::unique_ptr<IStructure> structure,
                            bool sort);
    void init() override;
    void shutdown() override;

    bool finalise_is_noop() const override { return m_mode == OutputMode::FASTQ || !m_sort; };

private:
    const std::unique_ptr<IStructure> m_structure;
    const bool m_sort;
    std::unordered_map<fs::path, std::unique_ptr<utils::HtsFile>> m_hts_files;

    void handle(const HtsData& data) override;
};

}  // namespace hts_writer

}  // namespace dorado
