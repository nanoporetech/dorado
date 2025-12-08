#pragma once

#include "hts_utils/hts_types.h"
#include "interface.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace dorado {
class HtsData;

namespace hts_writer {

class SummaryFileWriter : public IWriter {
public:
    using FieldFlags = uint32_t;
    static constexpr FieldFlags BASECALLING_FIELDS = 1 << 0;
    static constexpr FieldFlags POLYA_FIELDS = 1 << 1;
    static constexpr FieldFlags EXPERIMENT_FIELDS = 1 << 2;
    static constexpr FieldFlags BARCODING_FIELDS = 1 << 3;
    static constexpr FieldFlags ALIGNMENT_FIELDS = 1 << 4;
    static constexpr FieldFlags DUPLEX_FIELDS = 1 << 5;

    using AlignmentCounts = std::unordered_map<std::string, std::array<int, 3>>;

    SummaryFileWriter(const std::filesystem::path& output_directory,
                      FieldFlags flags,
                      std::optional<AlignmentCounts> alignment_counts);
    SummaryFileWriter(std::ostream& stream,
                      FieldFlags flags,
                      std::optional<AlignmentCounts> alignment_counts);

    void set_header(SamHdrSharedPtr header);
    void process(const Processable item) override;
    void shutdown() override;

    std::string get_name() const override { return "SummaryFileWriter"; }
    stats::NamedStats sample_stats() const override { return {}; }

private:
    void init();
    void handle(const HtsData& item) const;

    SamHdrSharedPtr m_shared_header{nullptr};
    std::unordered_map<std::string, dorado::ReadGroup> m_read_groups;
    int m_minimum_qscore{0};

    const std::optional<AlignmentCounts> m_alignment_counts;

    const FieldFlags m_field_flags;
    std::ofstream m_summary_file;
    std::ostream& m_summary_stream;
};

}  // namespace hts_writer
}  // namespace dorado
