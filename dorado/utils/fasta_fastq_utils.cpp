#include "fasta_fastq_utils.h"

namespace dorado::utils {

bool validate_sequence_and_replace_us(std::string& field) {
    bool contains_t{};
    bool contains_u{};
    for (auto& element : field) {
        switch (element) {
        case 'A':
        case 'C':
        case 'G':
            break;
        case 'T':
            if (contains_u) {
                return false;
            }
            contains_t = true;
            break;
        case 'U':
            if (contains_t) {
                return false;
            }
            contains_u = true;
            element = 'T';
            break;
        default:
            return false;
        }
    }

    return true;
}

bool get_non_empty_line(std::istream& input_stream, std::string& line) {
    if (!std::getline(input_stream, line)) {
        return false;
    }
    return !line.empty();
}

FastaFastqHeader::FastaFastqHeader(Format format) {
    switch (format) {
    case FASTA:
        m_start_char = '>';
        break;
    case FASTQ:
        m_start_char = '@';
        break;
    default:
        throw std::runtime_error("Invalid header format specifier");
    }
}

bool FastaFastqHeader::is_valid_id_field(const std::string& field) {
    if (field.size() < 2 || field.at(0) != m_start_char) {
        return false;
    }

    const auto id_start_char = field.at(1);
    if (id_start_char == ' ' || id_start_char == '\t') {
        return false;
    }

    return true;
}

void FastaFastqHeader::tokenize() {
    std::istringstream header_stream{m_header};

    std::string token;
    while (std::getline(header_stream, token, m_header_separator)) {
        m_tokens.push_back(std::move(token));
    }
}

bool FastaFastqHeader::set_header(std::string line) {
    if (!is_valid_id_field(line)) {
        return false;
    }
    m_header = std::move(line);
    auto delim_pos = m_header.find_first_of(" \t");
    if (delim_pos != std::string::npos) {
        m_header_separator = m_header[delim_pos];
    }
    tokenize();
    return true;
}

}  // namespace dorado::utils
