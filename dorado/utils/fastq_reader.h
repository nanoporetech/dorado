#pragma once

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::utils {

class FastqRecord {
public:
    // The header will be at least 2 characters and begin with '@'
    const std::string& header() const;

    // All characters will be in the set {A,C,G,T}
    const std::string& sequence() const;

    // All characters will be in the range of printable characters ['!' .. '~'],
    // i.e. [33 .. 126]
    const std::string& qstring() const;

    std::string_view read_id_view() const;
    std::string_view run_id_view() const;
    std::vector<std::string> get_bam_tags() const;

    //  Testability. Public to allow utests to check processing of tags embedded in the header record
    bool set_header(std::string value);

    static std::optional<FastqRecord> try_create(std::istream& input_stream,
                                                 std::string& error_message);

private:
    std::string m_header;
    std::string m_sequence;
    std::string m_qstring;

    bool m_header_has_bam_tags{};
    std::size_t token_len(std::size_t token_start_pos) const;
};

class FastqReader {
public:
    FastqReader(const std::string input_file);
    FastqReader(std::unique_ptr<std::istream> input_stream);
    bool is_valid() const;
    std::optional<FastqRecord> try_get_next_record();

private:
    std::string m_input_file;
    std::unique_ptr<std::istream> m_input_stream{};
    std::size_t m_record_count{};
};

// Check for a fastq file. Does basic checks on the four fields of the first record
// will return true for a sequence containing Us instead of Ts, so if the check
// succeeds it is still possible the file cannot be opened by HtsLib
bool is_fastq(const std::string& input_file);
bool is_fastq(std::istream& input_stream);

}  // namespace dorado::utils