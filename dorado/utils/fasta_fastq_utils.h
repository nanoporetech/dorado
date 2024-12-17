#pragma once

#include <istream>
#include <sstream>
#include <string>
#include <vector>

namespace dorado::utils {

bool validate_sequence_and_replace_us(std::string& field);

bool get_non_empty_line(std::istream& input_stream, std::string& line);

class FastaFastqHeader {
public:
    enum Format { FASTA, FASTQ };

    FastaFastqHeader(Format format);

    // The header will be at least 2 characters and begin with either '>' or '@'
    const std::string& header() const { return m_header; }

    // Tokenizes the header.
    // If the id field is followed by a '\t', then we tokenize by '\t', and
    // we assume all tokens after the first are BAM-style tokens, as emitted by
    // the fastq output of dorado.
    // If not, then we tokenize by spaces. This is useful if header contains
    // key-value pairs of the form "key=value" (as minknow does).
    const std::vector<std::string>& get_tokens() const { return m_tokens; };

    bool has_bam_tags() const { return m_header_separator == '\t'; }

    bool set_header(std::string value);

private:
    std::string m_header;
    char m_start_char;
    std::vector<std::string> m_tokens{};
    char m_header_separator{0};
    void tokenize();
    bool is_valid_id_field(const std::string& field);
};

}  // namespace dorado::utils
