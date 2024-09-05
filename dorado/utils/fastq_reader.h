#pragma once

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::utils {

class FastqRecord {
    std::string m_header;
    std::string m_sequence;
    std::string m_qstring;

    bool m_header_has_bam_tags{};
    std::size_t token_len(std::size_t token_start_pos) const;

public:
    const std::string& header() const;
    const std::string& sequence() const;
    const std::string& qstring() const;

    std::string_view read_id_view() const;
    std::string_view run_id_view() const;
    std::vector<std::string> get_bam_tags() const;

    bool set_id(std::string value);
    bool set_sequence(std::string value);
    bool set_quality(std::string value);
};

class FastqReader {
public:
    FastqReader(const std::string& input_file);
    FastqReader(std::unique_ptr<std::istream> input_stream);
    bool is_valid() const;
    std::optional<FastqRecord> try_get_next_record();

private:
    std::unique_ptr<std::istream> m_input_stream{};
};

// Check for a fastq file. Does basic checks on the four fields of the first record
// will return true for a sequence containing Us instead of Ts, so if the check
// succeeds it is still possible the file cannot be opened by HtsLib
bool is_fastq(const std::string& input_file);
bool is_fastq(std::istream& input_stream);

}  // namespace dorado::utils