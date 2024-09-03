#pragma once

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace dorado::utils {

class FastqRecord {
    std::string m_id;
    std::string m_sequence;
    std::string m_separator;
    std::string m_quality;

public:
    const std::string& id() const;
    const std::string& sequence() const;
    const std::string& separator() const;
    const std::string& quality() const;

    std::string_view read_id();

    bool set_id(std::string value);
    bool set_sequence(std::string value);
    bool set_separator(std::string value);
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