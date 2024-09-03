#include "fastq_reader.h"

#include <algorithm>
#include <cassert>
#include <fstream>

namespace dorado::utils {

namespace {

bool is_valid_id_field(const std::string& field) {
    if (field.at(0) != '@') {
        return false;
    }
    if (field.size() < 2 || field.at(1) == ' ') {
        return false;
    }
    return true;
}

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

bool is_valid_separator_field(const std::string& field) { return field == "+"; }

bool is_valid_quality_field(const std::string& field) {
    //0x21 (lowest quality; '!' in ASCII) to 0x7e (highest quality; '~' in ASCII)
    return std::none_of(field.begin(), field.end(), [](char c) { return c < 0x21 || c > 0x7e; });
}

bool get_non_empty_line(std::istream& input_stream, std::string& line) {
    if (!std::getline(input_stream, line)) {
        return false;
    }
    return !line.empty();
}

std::optional<FastqRecord> get_next_record(std::istream& input_stream) {
    if (!input_stream.good()) {
        return std::nullopt;
    }
    FastqRecord result;
    std::string line;
    if (!get_non_empty_line(input_stream, line) || !result.set_id(std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !result.set_sequence(std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !is_valid_separator_field(line)) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !result.set_quality(std::move(line))) {
        return std::nullopt;
    }

    if (result.sequence().size() != result.quality().size()) {
        return std::nullopt;
    }

    return result;
}

}  // namespace

const std::string& FastqRecord::id() const { return m_id; }
const std::string& FastqRecord::sequence() const { return m_sequence; }
const std::string& FastqRecord::quality() const { return m_quality; }

const std::string& FastqRecord::read_id() const { return m_read_id; }

bool FastqRecord::set_id(std::string line) {
    if (!is_valid_id_field(line)) {
        return false;
    }
    m_id = std::move(line);
    parse_id_line();
    return true;
}

bool FastqRecord::set_sequence(std::string line) {
    if (!validate_sequence_and_replace_us(line)) {
        return false;
    }
    m_sequence = std::move(line);
    return true;
}

bool FastqRecord::set_quality(std::string line) {
    if (!is_valid_quality_field(line)) {
        return false;
    }
    m_quality = std::move(line);
    return true;
}

void FastqRecord::parse_id_line() {
    assert(m_id.size() > 1);
    m_read_id = m_id.substr(1, m_id.find(' ') - 1);
}

FastqReader::FastqReader(const std::string& input_file) {
    if (!is_fastq(input_file)) {
        return;
    }
    m_input_stream = std::make_unique<std::ifstream>(input_file);
}

FastqReader::FastqReader(std::unique_ptr<std::istream> input_stream) {
    if (!is_fastq(*input_stream)) {
        return;
    }
    // return to start of stream after validating the first record.
    input_stream->clear();
    input_stream->seekg(0);
    m_input_stream = std::move(input_stream);
}

bool FastqReader::is_valid() const { return m_input_stream && m_input_stream->good(); }

std::optional<FastqRecord> FastqReader::try_get_next_record() {
    if (!m_input_stream) {
        return std::nullopt;
    }
    return get_next_record(*m_input_stream);
}

bool is_fastq(const std::string& input_file) {
    std::ifstream input_stream{input_file};
    return is_fastq(input_stream);
}

bool is_fastq(std::istream& input_stream) {
    if (!input_stream.good()) {
        return false;
    }

    return get_next_record(input_stream).has_value();
}

}  // namespace dorado::utils
